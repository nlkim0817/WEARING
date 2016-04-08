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
def tf( X, npx ):
    assert X[ 0 ].shape == ( npx, npx, 3 ) or X[ 0 ].shape == ( 3, npx, npx )
    #pdb.set_trace()
    if X[ 0 ].shape == ( npx, npx, 3 ):
        #X = X.transpose( 0, 3, 1, 2 )
        #pdb.set_trace()
	#color_grid_vis( X[[1],:,:,:], (1,1), os.path.join('3.png'))
	#pdb.set_trace()
	X = X.astype('float32')
	#pdb.set_trace()
	X[:,:,:,0] /= 50
        X[:,:,:,1] /= 127.5
        X[:,:,:,2] /= 127.5
        X = X.transpose( 0, 3, 1, 2 )
	#pdb.set_trace()
    return floatX( X  - 1. )

def itf( X, npx, mode='RGB' ):
    #pdb.set_trace()
    X = ( X.reshape( -1, nc, npx, npx ).transpose( 0, 2, 3, 1 ) + 1. ) / 2.
    #pdb.set_trace()
    if mode == 'LAB':
       #pdb.set_trace()
       X[:,:,:,0] *= 100
       X[:,:,:,1] *= 255
       X[:,:,:,2] *= 255
       X[:,:,:,1] -= 128
       X[:,:,:,2] -= 128
       #pdb.set_trace()
       
       for i in range(X.shape[0]):
         
	 #X[i,:,:,:]  = color.lab2rgb(X[i,])
	 #pdb.set_trace()
	 X[i,:,:,:]  = color.lab2rgb(X[i,:,:,:].astype('int8'))
         #pdb.set_trace()
    #pdb.set_trace()
    #X *= 255
    return X.astype('float32')

def ConvertGenInput( X_src, X_tar ):
    #X_src : person
    #X_tar : cloth
    #result : X_src(L-channel) + X_tar[AB-channel] : 3 dimensions
    #pdb.set_trace()
    Inp = np.concatenate( (X_src[:,[0],:,:], X_tar[:,[1,2],:,:]), axis = 1 )
    
    Gnd = X_src
    return Inp, Gnd
def ConvertGenOutput( X_src, X_est ): 
    #X_est : estimated AB-channel
    if X_est.shape[1] == 3:
       LAB_est = np.concatenate((X_src[:,[0],:,:], X_est[:,[1,2],:,:]), axis = 1 )
    if X_est.shape[1] == 2:
       LAB_est = np.concatenate((X_src[:,[0],:,:], X_est), axis = 1)
    #pdb.set_trace()
    #pdb.set_trace()
    return LAB_est
    # RGB_est = color.lab2rgb(LAB_est)
    # return floatX( RGB_est/ 0.5 - 1. )
def MakeVisual( X_src, X_tar): 
    #LAB pair
    #pdb.set_trace()
    #X_rst = np.zeros( X_src.shape, np.float32)
    #for i in range( X_src.shape[0]):
    #    X_rst[i,:,:,:] = np.concatenate(
    #	                 (np.resize( X_src[i,:,:,:], (1,nc,npx,npx/2)),
    #                      np.resize( X_tar[i,:,:,:], (1,nc,npx,npx/2))), axis =3 )


    X_src = np.resize(X_src,(X_src.shape[0],nc,npx,npx/2))
    X_tar = np.resize(X_tar,(X_tar.shape[0],nc,npx,npx/2))

    return X_tar
    #return np.concatenate( (X_src,X_tar), axis = 2) 


# SET PARAMETERS.
l2 = 1e-5           # l2 weight decay.
b1 = 0.5            # momentum term of adam.
nz = 64 	    # # dim of central activation of converter.
nc = 3              # # of channels in image.
batch_size = 128    # # of examples in batch.
npx = 64            # # of pixels width/height of input images.
nf = 128            # Primary # of filters.
nvis = 14           # # of samples to visualize during training.
niter_lr0 = 50      # # of iter at starting learning rate.
niter = 50          # # of total iteration.
lr_decay = 10       # # of iter to linearly decay learning rate to zero.
lr = 0.0002         # Initial learning rate for adam.

# INITIALIZE AND DEFINE CONVERTER-ENCODER.
relu = activations.Rectify(  )
sigmoid = activations.Sigmoid(  )
lrelu = activations.LeakyRectify(  )
tanh = activations.Tanh(  )
bce = T.nnet.binary_crossentropy
filt_ifn = inits.Normal( scale = 0.02 )
gain_ifn = inits.Normal( loc = 1., scale = 0.02 )
bias_ifn = inits.Constant( c = 0. )
ce_w1 = filt_ifn( ( nf, nc, 5, 5 ), 'ce_w1' )
ce_w2 = filt_ifn( ( nf * 2, nf, 5, 5 ), 'ce_w2' )
ce_g2 = gain_ifn( ( nf * 2 ), 'ce_g2' )
ce_b2 = bias_ifn( ( nf * 2 ), 'ce_b2' )
ce_w3 = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'ce_w3' )
ce_g3 = gain_ifn( ( nf * 4 ), 'ce_g3' )
ce_b3 = bias_ifn( ( nf * 4 ), 'ce_b3' )
ce_w4 = filt_ifn( ( nf * 8, nf * 4, 5, 5), 'ce_w4' )
ce_g4 = gain_ifn( ( nf * 8 ), 'ce_g4' )
ce_b4 = bias_ifn( ( nf * 8 ), 'ce_b4' )
ce_w5 = filt_ifn( ( nz, nf * 8, 4, 4 ), 'ce_w5' )
ce_g5 = gain_ifn( nz, 'ce_g5' )
ce_b5 = bias_ifn( nz, 'ce_b5' )
encoder_params = [ ce_w1, ce_w2, ce_g2, ce_b2, ce_w3, ce_g3, ce_b3, ce_w4, ce_g4, ce_b4, ce_w5, ce_g5, ce_b5 ]
def encoder( s, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5 ):
    h1 = lrelu( dnn_conv( s, w1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    z = lrelu( batchnorm( dnn_conv( h4, w5, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5, b = b5 ) )
    return T.flatten( z, 2 )

# INITIALIZE AND DEFINE CONVERTER-DECODER.
cd_w1 = filt_ifn( ( nz, nf * 8 * 4 * 4 ), 'cd_w1')
cd_g1 = gain_ifn( nf * 8 * 4 * 4, 'cd_g1' )
cd_b1 = bias_ifn( nf * 8 * 4 * 4, 'cd_b1' )
cd_w2 = filt_ifn( ( nf * 8, nf * 4, 5, 5 ), 'cd_w2' )
cd_g2 = gain_ifn( ( nf * 4 ), 'cd_g2' )
cd_b2 = bias_ifn( ( nf * 4 ), 'cd_b2' )
cd_w3 = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'cd_w3' )
cd_g3 = gain_ifn( ( nf * 2 ), 'cd_g3' )
cd_b3 = bias_ifn( ( nf * 2 ), 'cd_b3' )
cd_w4 = filt_ifn( ( nf * 2, nf, 5, 5 ), 'cd_w4' )
cd_g4 = gain_ifn( ( nf ), 'cd_g4' )
cd_b4 = bias_ifn( ( nf ), 'cd_b4' )
cd_w5 = filt_ifn( ( nf, nc-1, 5, 5 ), 'cd_w5' )
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
d_w1 = filt_ifn( ( nf, nc, 5, 5 ), 'd_w1' )
d_w2 = filt_ifn( ( nf * 2, nf, 5, 5 ), 'd_w2' )
d_g2 = gain_ifn( ( nf * 2 ), 'd_g2' )
d_b2 = bias_ifn( ( nf * 2 ), 'd_b2' )
d_w3 = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'd_w3' )
d_g3 = gain_ifn( ( nf * 4 ), 'd_g3' )
d_b3 = bias_ifn( ( nf * 4 ), 'd_b3' )
d_w4 = filt_ifn( ( nf * 8, nf * 4, 5, 5), 'd_w4' )
d_g4 = gain_ifn( ( nf * 8 ), 'd_g4' )
d_b4 = bias_ifn( ( nf * 8 ), 'd_b4' )
d_w5 = filt_ifn( ( nf * 8 * 4 * 4, 1 ), 'd_wy' )
discrim_params = [ d_w1, d_w2, d_g2, d_b2, d_w3, d_g3, d_b3, d_w4, d_g4, d_b4, d_w5 ]
def discrim( t, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5 ):
    h1 = lrelu( dnn_conv( t, w1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    yd = sigmoid( T.dot( T.flatten( h4, 2 ), w5 ) )
    return yd

# INITIALIZE AND DEFINE DOMAIN-DISCRIMINATOR.
"""
dd_w1 = filt_ifn( ( nf, (nc) * 2, 5, 5 ), 'dd_w1' )
dd_w2 = filt_ifn( ( nf * 2, nf, 5, 5 ), 'dd_w2' )
dd_g2 = gain_ifn( ( nf * 2 ), 'dd_g2' )
dd_b2 = bias_ifn( ( nf * 2 ), 'dd_b2' )
dd_w3 = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'dd_w3' )
dd_g3 = gain_ifn( ( nf * 4 ), 'dd_g3' )
dd_b3 = bias_ifn( ( nf * 4 ), 'dd_b3' )
dd_w4 = filt_ifn( ( nf * 8, nf * 4, 5, 5), 'dd_w4' )
dd_g4 = gain_ifn( ( nf * 8 ), 'dd_g4' )
dd_b4 = bias_ifn( ( nf * 8 ), 'dd_b4' )
dd_w5 = filt_ifn( ( nf * 16, nf * 8, 4, 4 ), 'dd_w5' )
dd_g5 = gain_ifn( ( nf * 16 ), 'dd_g5' )
dd_b5 = bias_ifn( ( nf * 16 ), 'dd_b5' )
dd_w6 = filt_ifn( ( nf * 16, 1 ), 'dd_w6' )
domain_discrim_params = [ dd_w1, dd_w2, dd_g2, dd_b2, dd_w3, dd_g3, dd_b3, dd_w4, dd_g4, dd_b4, dd_w5, dd_g5, dd_b5, dd_w6 ]
def domain_discrim( st, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6 ):
    h1 = lrelu( dnn_conv( st, w1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    h5 = lrelu( batchnorm( dnn_conv( h4, w5, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5, b = b5 ) )
    ydd = sigmoid( T.dot( T.flatten( h5, 2 ), w6 ) )
    return ydd
"""
# DEFINE FORWARD (COSTS) AND BACKWARD (UPDATE) FROM Z.
lrt = sharedX( lr )
IS = T.tensor4(  )
IT_sr = T.tensor4(  )
#IT_sf_con = T.tensor4( )
IT_sf = decoder( encoder( IS, *encoder_params ), *decoder_params )
IT_sru = T.tensor4(  )

#pdb.set_trace()

#IT_sf_con = 
IT_sf_con = T.concatenate( [IS[:,[0],:,:], IT_sf], axis = 1)
YD_sr = discrim( IT_sr, *discrim_params )
YD_sf = discrim( IT_sf_con, *discrim_params )
YD_sru = discrim( IT_sru, *discrim_params )

"""
IST_sr = T.concatenate( [ IS, IT_sr ], axis = 1 )
IST_sf = T.concatenate( [ IS, IT_sf ], axis = 1 )
IST_sru = T.concatenate( [ IS, IT_sru ], axis = 1 )
YDD_sr = domain_discrim( IST_sr, *domain_discrim_params )
YDD_sf = domain_discrim( IST_sf, *domain_discrim_params )
YDD_sru = domain_discrim( IST_sru, *domain_discrim_params )

# For converter."""
cost_d_for_ced_sf = bce( YD_sf, T.ones( YD_sf.shape ) ).mean(  )
#cost_dd_for_ced_sf = bce( YDD_sf, T.ones( YDD_sf.shape ) ).mean(  )
cost_mse = T.mean( ( IT_sr[:,[1,2],:,:] - IT_sf ) ** 2. )
#cost_for_ced_s = cost_d_for_ced_sf / 3. + cost_dd_for_ced_sf / 3. + cost_mse / 3.


#cost_for_ced_s = cost_d_for_ced_sf * 0.1 + cost_mse 
cost_for_ced_s = cost_d_for_ced_sf *0.01 + cost_mse
ced_s_updater = updates.Adam( lr = lrt, b1 = b1, regularizer = updates.Regularizer( l2 = l2 ) )
ced_s_updates = ced_s_updater( converter_params, cost_for_ced_s )

# For discriminator.
cost_d_for_d_sr = bce( YD_sr, T.ones( YD_sr.shape ) ).mean(  )
cost_d_for_d_sf = bce( YD_sf, T.zeros( YD_sf.shape ) ).mean(  )
cost_d_for_d_sru = bce( YD_sru, T.ones( YD_sru.shape ) ).mean(  )
cost_for_d_s = cost_d_for_d_sr / 3. + cost_d_for_d_sf / 3. + cost_d_for_d_sru / 3.
d_s_updater = updates.Adam( lr = lrt, b1 = b1, regularizer = updates.Regularizer( l2 = l2 ) )
d_s_updates = d_s_updater( discrim_params, cost_for_d_s )
"""
# For domain-discriminator.
cost_dd_for_dd_sr = bce( YDD_sr, T.ones( YDD_sr.shape ) ).mean(  )
cost_dd_for_dd_sf = bce( YDD_sf, T.zeros( YDD_sf.shape ) ).mean(  )
cost_dd_for_dd_sru = bce( YDD_sru, T.zeros( YDD_sru.shape ) ).mean(  )
cost_for_dd_s = cost_dd_for_dd_sr / 3. + cost_dd_for_dd_sf / 3. + cost_dd_for_dd_sru / 3.
dd_s_updater = updates.Adam( lr = lrt, b1 = b1, regularizer = updates.Regularizer( l2 = l2 ) )
dd_s_updates = dd_s_updater( domain_discrim_params, cost_for_dd_s )
"""
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
_train_ced_s = theano.function( [ IS, IT_sr ], cost_for_ced_s, updates = ced_s_updates )
_train_d_s = theano.function( [ IS, IT_sr, IT_sru ], cost_for_d_s, updates = d_s_updates )
#_train_dd_s = theano.function( [ IS, IT_sr, IT_sru ], cost_for_dd_s, updates = dd_s_updates )
_val_ced_s = theano.function( [ IS, IT_sr ], cost_for_ced_s )
_val_d_s = theano.function( [ IS, IT_sr, IT_sru ], cost_for_d_s )
#_val_dd_s = theano.function( [ IS, IT_sr, IT_sru ], cost_for_dd_s )
_test_ced = theano.function( [ IS ], IT_sf )
print '%.2f seconds to compile theano functions.' % ( time(  ) - t )

# PREPARE FOR DATAOUT.
dataout = os.path.join( './dataout_2/', di_st.name )
desc = ( sys.argv[ 0 ][ 0 : -3 ] ).upper(  )
model_dir = os.path.join( dataout, desc, 'models'.upper(  ) )
sample_dir = os.path.join( dataout, desc, 'samples'.upper(  ) )
if not os.path.exists( model_dir ):
    os.makedirs( model_dir )
if not os.path.exists( sample_dir ):
    os.makedirs( sample_dir )

# PLOT SOURCE/TARGET SAMPLE IMAGES.
np_rng = np.random.RandomState( 1 )
vis_tr = np_rng.permutation( len( sset_tr ) )
vis_tr = vis_tr[ 0 : nvis ** 2 ]
vis_ims_tr_s = ims_LAB[ sset_tr[ vis_tr ] ]
vis_ims_tr_t = ims_LAB[ tset_tr[ vis_tr ] ]
vis_tr_input, vis_tr_gnd = ConvertGenInput(vis_ims_tr_s, vis_ims_tr_t)
vis_ims_tr_t_hat = _test_ced( vis_tr_input )

#pdb.set_trace()
#color_grid_vis( itf( vis_ims_tr_s ,npx, 'LAB'),
#                ( nvis, nvis),
#		os.path.join( sample_dir, 'TR_S1.png') )
#color_grid_vis( itf( vis_ims_tr_t ,npx, 'LAB'),
#                ( nvis, nvis),
#		os.path.join( sample_dir, 'TR_S2.png') )
color_grid_vis( itf( MakeVisual( vis_ims_tr_s, vis_ims_tr_t  ), npx, 'LAB'),
                ( nvis, nvis),
		os.path.join( sample_dir, 'TR_S.png') )

color_grid_vis( itf( ConvertGenOutput( vis_tr_gnd, vis_ims_tr_s),npx,'LAB' ),
                (nvis, nvis),
		os.path.join( sample_dir, 'TR_T.png') )
color_grid_vis( itf( ConvertGenOutput(vis_ims_tr_s, vis_ims_tr_t_hat),npx,'LAB' ),
                (nvis, nvis),
		os.path.join( sample_dir, 'TR000T.png') )

vis_val = np_rng.permutation( len( sset_val ) )
vis_val = vis_val[ 0 : nvis ** 2 ]
vis_ims_val_s = ims_LAB[ sset_val[ vis_val ] ]
vis_ims_val_t = ims_LAB[ tset_val[ vis_val ] ]
vis_val_input, vis_val_gnd  = ConvertGenInput(vis_ims_val_s, vis_ims_val_t)
vis_ims_val_t_hat = _test_ced( vis_val_input )

color_grid_vis( itf( MakeVisual(vis_ims_val_s, vis_ims_val_t ),npx,'LAB' ),
                ( nvis, nvis),
		os.path.join( sample_dir, 'VAL_S.png') )
color_grid_vis( itf( ConvertGenOutput( vis_val_gnd, vis_ims_val_s),npx, 'LAB' ),
                (nvis, nvis),
		os.path.join( sample_dir, 'VAL_T.png') )

color_grid_vis( itf( ConvertGenOutput(vis_ims_val_s, vis_ims_val_t_hat),npx, 'LAB'),
                (nvis, nvis),
		os.path.join( sample_dir, 'VAL000T.png') )

# DO THE JOB.
print desc.upper(  )
num_epoch = 0
t = time(  )
for epoch in range( niter ):
    # Decay learning rate if needed.
    num_epoch += 1
    np_rng = np.random.RandomState( num_epoch ) # Neccessary for same result for the case of loaded network.
    if num_epoch > niter_lr0:
        print( 'Decaying learning rate.' )
        lrt.set_value( floatX( lrt.get_value(  ) - lr / lr_decay ) )
    # Load pre-trained param if exists.
    mpath_ce = os.path.join( model_dir, 'CE%03d.npy' % num_epoch )
    mpath_cd = os.path.join( model_dir, 'CD%03d.npy' % num_epoch )
    mpath_d = os.path.join( model_dir, 'D%03d.npy' % num_epoch )
    #mpath_dd = os.path.join( model_dir, 'DD%03d.npy' % num_epoch )
    if os.path.exists( mpath_ce ) and os.path.exists( mpath_cd ) and os.path.exists( mpath_d ):
    #and os.path.exists( mpath_dd ):
        print( 'Epoch %02d: Load.' % num_epoch )
        data_ce = np.load( mpath_ce )
        data_cd = np.load( mpath_cd )
        data_d = np.load( mpath_d )
        #data_dd = np.load( mpath_dd )
        for pi in range( len( encoder_params ) ):
            encoder_params[ pi ].set_value( data_ce[ pi ] )
        for pi in range( len( decoder_params ) ):
            decoder_params[ pi ].set_value( data_cd[ pi ] )
        for pi in range( len( discrim_params ) ):
            discrim_params[ pi ].set_value( data_d[ pi ] )
        #for pi in range( len( domain_discrim_params ) ):
        #    domain_discrim_params[ pi ].set_value( data_dd[ pi ] )
        continue
    # Training.
    num_sample_st = len( sset_tr )
    num_batch = int( np.ceil( num_sample_st / float( batch_size ) ) )
    cost_for_ced_s_cumm = 0.
    cost_for_d_s_cumm = 0.
    #cost_for_dd_s_cumm = 0.
    for bi in range( num_batch ):
        # Define batch.
        bis = bi * batch_size
        bie = min( bi * batch_size + batch_size, num_sample_st )
        this_bsize = bie - bis
        Pb = pset_tr[ bis : bie ]
        ISb = ims_LAB[ sset_tr[ bis : bie ] ] #person
        ITb_sr = ims_LAB[ tset_tr[ bis : bie ] ] #cloth
        	
	ITb_sru = np.zeros( ISb.shape, ISb.dtype )
        for b in range( this_bsize ):
            iid = sset_tr[ np_rng.choice( ( pset_tr != Pb[ b ] ).nonzero(  )[ 0 ], 1 ) ]
            ITb_sru[ b ] = ims_LAB[ iid ]
        # Flip augmentation.
        for b in range( this_bsize ):
            if np_rng.uniform(  ) > .5:
                ISb[ b ] = ( np.fliplr( ISb[ b ].transpose( 1, 2, 0 ) ) ).transpose( 2, 0, 1 )
            if np_rng.uniform(  ) > .5:
                ITb_sr[ b ] = ( np.fliplr( ITb_sr[ b ].transpose( 1, 2, 0 ) ) ).transpose( 2, 0, 1 )
            if np_rng.uniform(  ) > .5:
                ITb_sru[ b ] = ( np.fliplr( ITb_sru[ b ].transpose( 1, 2, 0 ) ) ).transpose( 2, 0, 1 ) 
        ISb_input,ISb_output = ConvertGenInput(ISb, ITb_sr)
	"""# Train converter."""
        cost_for_ced_s = _train_ced_s( ISb_input, ISb_output ) # Update ced * 2
        cost_for_ced_s_cumm += cost_for_ced_s * this_bsize
        
	"""# Train discriminator."""
        cost_for_d_s = _train_d_s( ISb_input, ISb_output, ITb_sru ) # Update d * 2
        cost_for_d_s_cumm += cost_for_d_s * this_bsize
        # Train domain-discriminator.
        #cost_for_dd_s = _train_dd_s( ISb, ITb_sr, ITb_sru ) # Update dd * 2
        #cost_for_dd_s_cumm += cost_for_dd_s * this_bsize
        # Monitor.
        if np.mod( bi, num_batch / 20 ) == 0:
            prog = np.round( bi * 100. / num_batch )
            print( 'Epoch %d: ced_s = %.4f, d_s = %.4f %d%% (%d/%d)' 
                    % ( num_epoch, cost_for_ced_s, cost_for_d_s,  prog, bi + 1, num_batch ) )
    # Save network.
    cost_for_ced_s_cumm /= num_sample_st
    cost_for_d_s_cumm /= num_sample_st
    #cost_for_dd_s_cumm /= num_sample_st
    print( 'Epoch %d: ced_s = %.4f, d_s = %.4f' 
            % ( num_epoch, cost_for_ced_s_cumm, cost_for_d_s_cumm ) )
    print( 'Epoch %d: Save.' % num_epoch )
    np.save( mpath_ce, [ p.get_value(  ) for p in encoder_params ] )
    np.save( mpath_cd, [ p.get_value(  ) for p in decoder_params ] )
    np.save( mpath_d, [ p.get_value(  ) for p in discrim_params ] )
    #np.save( mpath_dd, [ p.get_value(  ) for p in domain_discrim_params ] )
    # Validation.
    num_sample_st = len( sset_val )
    num_batch = int( np.ceil( num_sample_st / float( batch_size ) ) )
    cost_for_ced_s_cumm = 0.
    cost_for_d_s_cumm = 0.
    #cost_for_dd_s_cumm = 0.
    for bi in range( num_batch ):
        # Define batch.
        bis = bi * batch_size
        bie = min( bi * batch_size + batch_size, num_sample_st )
        this_bsize = bie - bis
        Pb = pset_val[ bis : bie ]
        ISb = ims_LAB[ sset_val[ bis : bie ] ]
        ITb_sr = ims_LAB[ tset_val[ bis : bie ] ]
        ITb_sru = np.zeros( ISb.shape, ISb.dtype )
        
	for b in range( this_bsize ):
            iid = sset_val[ np_rng.choice( ( pset_val != Pb[ b ] ).nonzero(  )[ 0 ], 1 ) ]
            ITb_sru[ b ] = ims_LAB[ iid ]
        
        ISb_input,ISb_output = ConvertGenInput(ISb, ITb_sr)
 
	"""# Val converter."""
        cost_for_ced_s = _val_ced_s( ISb_input, ISb_output )
        cost_for_ced_s_cumm += cost_for_ced_s * this_bsize
        """# Val discriminator."""
        cost_for_d_s = _val_d_s( ISb_input, ISb_output, ITb_sru )
        cost_for_d_s_cumm += cost_for_d_s * this_bsize
        
	# Val domain-discriminator.
        #cost_for_dd_s = _val_dd_s( ISb, ITb_sr, ITb_sru )
        #cost_for_dd_s_cumm += cost_for_dd_s * this_bsize
        # Monitor.
        if np.mod( bi, num_batch / 20 ) == 0:
            prog = np.round( bi * 100. / num_batch )
            print( 'Val) Epoch %d: %d%% (%d/%d)' % ( num_epoch, prog, bi + 1, num_batch ) )
    cost_for_ced_s_cumm /= num_sample_st
    cost_for_d_s_cumm /= num_sample_st
    #cost_for_dd_s_cumm /= num_sample_st
    print( 'Val) Epoch %d: ced_s = %.4f, d_s = %.4f' 
            % ( num_epoch, cost_for_ced_s_cumm, cost_for_d_s_cumm ) )
    # Sample visualization.
    

    samples =  _test_ced( vis_tr_input ) 
    
    color_grid_vis( itf( ConvertGenOutput( vis_tr_gnd, samples), npx, 'LAB' ), 
            ( nvis, nvis ),
            os.path.join( sample_dir, 'TR%03dT.png' % num_epoch ) )

    samples = _test_ced( vis_val_input ) 
    color_grid_vis( itf( ConvertGenOutput( vis_val_gnd, samples), npx, 'LAB' ), 
            ( nvis, nvis ),
            os.path.join( sample_dir, 'VAL%03dT.png' % num_epoch ) )

