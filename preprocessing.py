import os.path
DatasetName='UCF101'
mainpath = '/data/lisatmp3/negar/Datasets/'+DatasetName+'/'
downsampledimages = '/Tmp/rostamn/'+'UCF101'
if not os.path.exists(downsampledimages):
	os.makedirs(downsampledimages)
i = 0
import ipdb; ipdb.set_trace()
allcategories=os.listdir(mainpath)
for j in range(len(allcategories)):
    for root, dirs, files in os.walk(mainpath+allcategories[j]+'/'):
        print files
    videopath=mainpath+allcategories[j]+'/'
    text_file = open("imagextr.txt", "w")
    ctr=0
    for i in range(len(files)):
        ctr=ctr+1
        print str(ctr)+' out of '+str(13320)+' ******************************************'
	vidname = files[i]
       
      #  if vidname!='s17-d13-cam-002.avi':
       #     continue
        if not os.path.exists(downsampledimages+'/'+allcategories[j]):
            os.makedirs(downsampledimages+'/'+allcategories[j])
	if not os.path.exists(downsampledimages+'/'+allcategories[j]+'/'+vidname[:-4]):
	    os.makedirs(downsampledimages+'/'+allcategories[j]+'/'+vidname[:-4])
        print vidname[:-4]
	#os.system("ffmpeg -i "+videopath+vidname+" "+downsampledimages+'/'+vidname[:-4]+"/"+"image%d.jpg")
	#print "ffmpeg -i "+videopath+vidname+" "+downsampledimages+'/'+vidname[:-4]+"/"+"image%d.jpg"
        #strcmd="/data/lisatmp3/yaoli/caffe/feature_extraction/ffmpeg/bin/ffmpeg -qscale 2 -vf crop=1150:735:304:200,scale=320:200 -i "+ videopath+vidname+"  "+downsampledimages+'/'+vidname[:-4] +"/image-%d.png\n" 
        strcmd="/data/lisatmp3/yaoli/caffe/feature_extraction/ffmpeg/bin/ffmpeg -qscale 2 -vf scale=240:240 -i "+videopath+vidname+"  "+downsampledimages+'/'+allcategories[j]+'/'+vidname[:-4] +"/image-%d.png\n" 
 
        os.system(strcmd)
        print strcmd
	text_file.write(strcmd)
		
text_file.close()

  
