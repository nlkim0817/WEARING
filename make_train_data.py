import os
from glob import glob
import numpy as np
import scipy.misc
import math
# Set params.
src_dir = '/iron/db/celebA/'
dst_dir = './db/celebA/'
im_format = '*.jpg'
im_size = 64
keep_aspect = True
# Do the job.
if not os.path.exists( dst_dir ):
	os.mkdir( dst_dir )
iid2impath = glob( os.path.join( src_dir, im_format ) )
num_im = len( iid2impath )
print 'num im: %d' % num_im
for i in range( num_im ):
	print 'im %06d / %06d' % ( i, num_im ),
	impath = iid2impath[ i ]
	im = scipy.misc.imread( impath )
	nr, nc, nch = im.shape	
	print 'from (%d,%d,%d)' % ( nr, nc, nch ),
	if nr >= nc:
		margin = math.floor( float( nr - nc ) / 2.0 )
		rs = margin
		re = margin + nc
		imc = im[ rs : re, :, : ]
	else:
		margin = math.floor( float( nc - nr ) / 2.0 )
		cs = margin
		ce = margin + nr
		imc = im[ :, cs : ce, : ]
	imc = scipy.misc.imresize( imc, [ im_size, im_size, nch ] )
	nr, nc, nch = imc.shape
	print 'to (%d,%d,%d)' % ( nr, nc, nch )
	_, fname = os.path.split( impath )
	dst_path = os.path.join( dst_dir, fname )
	scipy.misc.imsave( dst_path, imc )
print 'Done.'
