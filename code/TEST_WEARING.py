import sys
import os
from time import time
import numpy as np
import theano
import theano.tensor as T
from skimage import color
from theano.sandbox.cuda.dnn import dnn_conv
sys.path.append( '..' )
from lib import activations, updates, inits
from lib.vis import color_grid_vis
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX
from Datain import Pldt, Gan
import pdb
np_rng = np.random.RandomState( 1 )
def tf( X, npx ):
    assert X[ 0 ].shape == ( npx, npx, 3 ) or X[ 0 ].shape == ( 3, npx, npx )
    if X[ 0 ].shape == ( npx, npx, 3 ):
	X = X.astype('float32')
	X[:,:,:,0] /= 50
        X[:,:,:,1] /= 127.5
        X[:,:,:,2] /= 127.5
        X = X.transpose( 0, 3, 1, 2 )
    return floatX( X  - 1. )
def itf( X, npx, mode='RGB' ):
    X = ( X.reshape( -1, nc, npx, npx ).transpose( 0, 2, 3, 1 ) + 1. ) / 2.
    if mode == 'LAB':
       X[:,:,:,0] *= 100
       X[:,:,:,1] *= 255
       X[:,:,:,2] *= 255
       X[:,:,:,1] -= 128
       X[:,:,:,2] -= 128
       for i in range(X.shape[0]):
	 X[i,:,:,:]  = color.lab2rgb(X[i,:,:,:].astype('int8'))
    return X.astype('float32')
def ConvertGenInput( X_src, X_tar ):
    Inp = np.concatenate( (X_src[:,[0],:,:], X_tar[:,[1,2],:,:]), axis = 1 )
    Gnd = X_src
    return Inp, Gnd
def ConvertGenOutput( X_src, X_est ): 
    if X_est.shape[1] == 3:
       LAB_est = np.concatenate((X_src[:,[0],:,:], X_est[:,[1,2],:,:]), axis = 1 )
    if X_est.shape[1] == 2:
       LAB_est = np.concatenate((X_src[:,[0],:,:], X_est), axis = 1)
    return LAB_est
def MakeVisual( X_src, X_tar): 
    X_src = np.resize(X_src,(X_src.shape[0],nc,npx,npx/2))
    X_tar = np.resize(X_tar,(X_tar.shape[0],nc,npx,npx/2))
    return X_tar

# SET PARAMETERS.
l2 = 1e-5           # l2 weight decay.
b1 = 0.5            # momentum term of adam.
nz = 64 	    # # dim of central activation of converter.
nc = 3              # # of channels in image.
batch_size = 196    # # of examples in batch.
npx = 64            # # of pixels width/height of input images.
nf = 128            # Primary # of filters.
nvis = 14           # # of samples to visualize during training.
niter_lr0 = 50      # # of iter at starting learning rate.
niter = 50          # # of total iteration.
lr_decay = 10       # # of iter to linearly decay learning rate to zero.
lr = 0.0002         # Initial learning rate for adam.



Load_num = 50
main_dir = './dataout_2/LOOKBOOK/WEARING/MODELS/'
ce_dir = os.path.join(main_dir,'CE0%02d.npy' % Load_num)
cd_dir = os.path.join(main_dir,'CD0%02d.npy' % Load_num)
d_dir  = os.path.join(main_dir,'CD0%02d.npy' % Load_num)

ce_data = np.load(ce_dir)
cd_data = np.load(cd_dir)
d_data  = np.load(d_dir)



# INITIALIZE AND DEFINE CONVERTER-ENCODER.
relu = activations.Rectify(  )
sigmoid = activations.Sigmoid(  )
lrelu = activations.LeakyRectify(  )
tanh = activations.Tanh(  )
bce = T.nnet.binary_crossentropy
filt_ifn = inits.Normal( scale = 0.02 )
gain_ifn = inits.Normal( loc = 1., scale = 0.02 )
bias_ifn = inits.Constant( c = 0. )
ce_w1 = sharedX(ce_data[0])
ce_w2 = sharedX(ce_data[1])
ce_g2 = sharedX(ce_data[2])
ce_b2 = sharedX(ce_data[3])
ce_w3 = sharedX(ce_data[4])
ce_g3 = sharedX(ce_data[5])
ce_b3 = sharedX(ce_data[6])
ce_w4 = sharedX(ce_data[7])
ce_g4 = sharedX(ce_data[8])
ce_b4 = sharedX(ce_data[9])
ce_w5 = sharedX(ce_data[10])
ce_g5 = sharedX(ce_data[11])
ce_b5 = sharedX(ce_data[12])
encoder_params = [ ce_w1, ce_w2, ce_g2, ce_b2, ce_w3, ce_g3, ce_b3, ce_w4, ce_g4, ce_b4, ce_w5, ce_g5, ce_b5 ]
def encoder( s, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5 ):
    h1 = lrelu( dnn_conv( s, w1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    z = lrelu( batchnorm( dnn_conv( h4, w5, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5, b = b5 ) )
    return T.flatten( z, 2 )

# INITIALIZE AND DEFINE CONVERTER-DECODER.
cd_w1 = sharedX(cd_data[0]) 
cd_g1 = sharedX(cd_data[1]) 
cd_b1 = sharedX(cd_data[2]) 
cd_w2 = sharedX(cd_data[3]) 
cd_g2 = sharedX(cd_data[4]) 
cd_b2 = sharedX(cd_data[5]) 
cd_w3 = sharedX(cd_data[6]) 
cd_g3 = sharedX(cd_data[7]) 
cd_b3 = sharedX(cd_data[8]) 
cd_w4 = sharedX(cd_data[9]) 
cd_g4 = sharedX(cd_data[10]) 
cd_b4 = sharedX(cd_data[11]) 
cd_w5 = sharedX(cd_data[12]) 
decoder_params = [ cd_w1, cd_g1, cd_b1, cd_w2, cd_g2, cd_b2, cd_w3, cd_g3, cd_b3, cd_w4, cd_g4, cd_b4, cd_w5 ]
def decoder( z, w1, g1, b1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5 ):
    h1 = relu( batchnorm( T.dot( z, w1 ), g = g1, b = b1 ) )
    h1 = h1.reshape( (h1.shape[ 0 ], nf * 8, 4, 4 ) ) 
    h2 = relu( batchnorm( deconv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = relu( batchnorm( deconv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = relu( batchnorm( deconv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    t = tanh( deconv( h4, w5, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ) )
    return t
converter_params = [ ce_w1, ce_w2, ce_g2, ce_b2, ce_w3, ce_g3, ce_b3, ce_w4, ce_g4, ce_b4, ce_w5, 
        cd_w1, cd_g1, cd_b1, cd_w2, cd_g2, cd_b2, cd_w3, cd_g3, cd_b3, cd_w4, cd_g4, cd_b4, cd_w5 ]

# INITIALIZE AND DEFINE DISCRIMINATOR.
d_w1 = sharedX(d_data[0])
d_w2 = sharedX(d_data[1])
d_g2 = sharedX(d_data[2])
d_b2 = sharedX(d_data[3])
d_w3 = sharedX(d_data[4])
d_g3 = sharedX(d_data[5])
d_b3 = sharedX(d_data[6])
d_w4 = sharedX(d_data[7])
d_g4 = sharedX(d_data[8])
d_b4 = sharedX(d_data[9])
d_w5 = sharedX(d_data[10])
discrim_params = [ d_w1, d_w2, d_g2, d_b2, d_w3, d_g3, d_b3, d_w4, d_g4, d_b4, d_w5 ]
def discrim( t, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5 ):
    h1 = lrelu( dnn_conv( t, w1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    yd = sigmoid( T.dot( T.flatten( h4, 2 ), w5 ) )
    return yd

# DEFINE FORWARD (COSTS) AND BACKWARD (UPDATE) FROM Z.
lrt = sharedX( lr )
IS = T.tensor4(  )
#IT_sf_con = T.tensor4( )
IT_sf = decoder( encoder( IS, *encoder_params ), *decoder_params )

# PREPARE FOR DATAIN.
di_st = Pldt(  )
di_st.set_LOOKBOOK(  )
ims_LAB = di_st.load( npx, True )
print( 'Swap dimension of %s.' % di_st.name )
ims_LAB = tf( ims_LAB, npx )
#img_RGB = tf( ims_RGB, npx )
print( 'Done.' )
di_st.shuffle(  )
sset_tr = di_st.d1set_tr
tset_tr = di_st.d2set_tr
pset_tr = di_st.pids_tr
sset_val = di_st.d1set_val
tset_val = di_st.d2set_val
pset_val = di_st.pids_val

#COMPILING TRAIN/TEST FUNCTIONS.
print 'COMPILING'
t = time(  )
_test_ced = theano.function( [ IS ], IT_sf )
print '%.2f seconds to compile theano functions.' % ( time(  ) - t )
# DO THE JOB.
numVal = 0
save_dir = './save/'
t = time(  )
num_batch = int( np.ceil( len( sset_val) / float( batch_size)))
for bi in range( num_batch ):
    bis = bi * batch_size
    bie = min( bi*batch_size + batch_size, len( sset_val) )
    this_bsize = bie - bis
    Pb = pset_val[ bis:bie ]
    ISb = ims_LAB[ sset_val[ bis:bie] ]
    ISb_sr = np.zeros( ISb.shape, ISb.dtype )
    for b in range (this_bsize):
        iid = tset_val[ np_rng.choice( ( pset_val != Pb[b] ).nonzero( )[ 0 ], 1) ]
	ISb_sr[b] = ims_LAB[ iid ]
    ISb_input,Gnd = ConvertGenInput(ISb, ISb_sr)
    results = _test_ced(ISb_input)
    numVal += 1
    color_grid_vis( itf( ConvertGenOutput( Gnd, results), npx, 'LAB' ),
                    ( nvis, nvis ),
		    os.path.join( save_dir, 'VAL%03d.png' %numVal))
    color_grid_vis( itf( ISb, npx, 'LAB'),
                    ( nvis, nvis ),
		    os.path.join( save_dir, 'PER%03d.png' %numVal))
    color_grid_vis( itf( ISb_sr, npx, 'LAB'),
                    ( nvis, nvis ),
		    os.path.join( save_dir, 'CLO%03d.png' %numVal))
    print( 'Test) Iteration : %d, (BATCHSIZE : %d of %d' %( numVal, bi, num_batch))

