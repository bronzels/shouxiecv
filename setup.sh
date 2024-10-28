if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Mac detected."
    #mac
    os=darwin
    MYHOME=/Volumes/data
    SED=gsed
    bin=/Users/apple/bin
else
    echo "Assuming linux by default."
    #linux
    os=linux
    MYHOME=~
    SED=sed
    bin=/usr/local/bin
fi

pip install jupyter notebook
pip install opencv-python opencv-contrib-python opencv-python-headless
#-4.10.0.84
pip install matplotlib 

git clone git@github.com:AccumulateMore/OpenCV
mkdir opencv
cd opencv
ln -s ../orig-OpenCV/01_Picture 01_Picture
ln -s ../orig-OpenCV/02_Video 02_Video

#https://www.haolizi.net/example/view_271111.html
#下载代码解压
mkdir template-matching-ocr
cd template-matching-ocr
ln -s ../orig-template-matching-ocr/images images
ln -s ../orig-template-matching-ocr/ocr_a_reference.png ocr_a_reference.png


#yum install tesseract -y
#3.04
#yum install tesseract-langpacks -y
#安装gcc/g++/make/automake
#安装leptonica
yum install -y ca-certificates libtool libtiff-devel libjpeg-devel libpng-devel
wget -c http://www.leptonica.org/source/leptonica-1.84.1.tar.gz 
tar zxf leptonica-1.84.1.tar.gz 
cd leptonica-1.84.1/
./configure && make -j 100 && make install
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib" >> ~/.bashrc
echo "export LIBLEPT_HEADERSDIR=/usr/local/include" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig" >> ~/.bashrc
source ~/.bashrc
#安装tesseract
wget -c https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.4.1.tar.gz
tar xzvf tesseract-5.4.1.tar.gz
cd tesseract-5.4.1
./autogen.sh
brew install cairo pango icu4c autoconf libffi libarchive libpng
export PKG_CONFIG_PATH="/usr/local/opt/icu4c/lib/pkgconfig"
pip install pyicu
./configure --prefix=/data0/tesseract --with-extra-includes=/usr/local/include --with-extra-libraries=/usr/local/include
#可能时间长
:<<EOF
configure: WARNING: icu 52.1 or higher is required, but was not found.
configure: WARNING: Training tools WILL NOT be built.
configure: WARNING: Try to install libicu-dev package.
checking for pango >= 1.38.0... no
configure: WARNING: pango 1.38.0 or higher is required, but was not found.
configure: WARNING: Training tools WILL NOT be built.
configure: WARNING: Try to install libpango1.0-dev package.
checking for cairo... yes
checking for pangocairo... no
checking for pangoft2... no
EOF
make -j 100
make install
make training -j 100
make training-install
ldconfig
echo "export PATH=\$PATH:/data0/tesseract/bin" >> ~/.bashrc
echo "export PATH=\$PATH:/data0/tesseract/bin" >> ~/.bashrc
source ~/.bashrc
tesseract --version
:<<EOF
tesseract 5.4.1
 leptonica-1.84.1
  libjpeg 6b (libjpeg-turbo 1.2.90) : libpng 1.5.13 : libtiff 4.0.3 : zlib 1.2.7
 Found AVX2
 Found AVX
 Found FMA
 Found SSE4.1
 Found OpenMP 201511
 Found libcurl/7.29.0 NSS/3.90 zlib/1.2.7 libidn/1.28 libssh2/1.8.0
EOF
cd /data0
cd /data0/tesseract/share
mv tessdata tessdata.bk
git clone git@github.com:tesseract-ocr/tessdata.git
#git clone git@github.com:tesseract-ocr/tessdata_fast.git tessdata
#git clone git@github.com:tesseract-ocr/tessdata_best.git tessdata
rm -f tessdata/configs
rm -rf tessdata/tessconfigs
rm -f tessdata/pdf.ttf
cp -r tessdata.bk/configs tessdata/
cp -r tessdata.bk/tessconfigs tessdata/
cp tessdata.bk/pdf.ttf tessdata/
tesseract --list-langs
:<<EOF
List of available languages in "/data0/tesseract/share/tessdata/" (166):
afr
amh
ara
asm
aze
aze_cyrl
bel
ben
bod
bos
bre
bul
cat
ceb
ces
chi_sim
chi_sim_vert
chi_tra
chi_tra_vert
chr
cos
cym
dan
dan_frak
deu
deu_frak
deu_latf
div
dzo
ell
eng
enm
epo
equ
est
eus
fao
fas
fil
fin
fra
frm
fry
gla
gle
glg
grc
guj
hat
heb
hin
hrv
hun
hye
iku
ind
isl
ita
ita_old
jav
jpn
jpn_vert
kan
kat
kat_old
kaz
khm
kir
kmr
kor
kor_vert
lao
lat
lav
lit
ltz
mal
mar
mkd
mlt
mon
mri
msa
mya
nep
nld
nor
oci
ori
osd
pan
pol
por
pus
que
ron
rus
san
script/Arabic
script/Armenian
script/Bengali
script/Canadian_Aboriginal
script/Cherokee
script/Cyrillic
script/Devanagari
script/Ethiopic
script/Fraktur
script/Georgian
script/Greek
script/Gujarati
script/Gurmukhi
script/HanS
script/HanS_vert
script/HanT
script/HanT_vert
script/Hangul
script/Hangul_vert
script/Hebrew
script/Japanese
script/Japanese_vert
script/Kannada
script/Khmer
script/Lao
script/Latin
script/Malayalam
script/Myanmar
script/Oriya
script/Sinhala
script/Syriac
script/Tamil
script/Telugu
script/Thaana
script/Thai
script/Tibetan
script/Vietnamese
sin
slk
slk_frak
slv
snd
spa
spa_old
sqi
srp
srp_latn
sun
swa
swe
syr
tam
tat
tel
tgk
tgl
tha
tir
ton
tur
uig
ukr
urd
uzb
uzb_cyrl
vie
yid
yor
EOF
cd ..
tesseract test.png output_1 -l eng
#real:762408
#output:162409
#参考以下用jTessBoxEditor重新xunli9antesseract
#https://blog.csdn.net/weixin_44143876/article/details/134485827
#https://sourceforge.net/projects/vietocr/files/jTessBoxEditor/jTessBoxEditor-2.6.0.zip/download
unzip jTessBoxEditor-2.6.0.zip
#打开jTessBoxEditor/jTessBoxEditor.jar，Tools->Merge TIFF，将样本文件全部选上，并将合并文件保存为numhw.font.exp0.tiff
mkdir newfont
cd newfont
tesseract numhw.font.exp0.tiff numhw.font.exp0 batch.nochop makebox
echo "font 0 0 0 0 0" >> font_properties
cat << EOF > numhw.sh
echo "Run Tesseract for Training.. "
tesseract numhw.font.exp0.tiff numhw.font.exp0 nobatch box.train 
 
echo "Compute the Character Set.. "
unicharset_extractor numhw.font.exp0.box
mftraining -F font_properties -U unicharset -O numhw.unicharset numhw.font.exp0.tr 


echo "Clustering.. "
cntraining numhw.font.exp0.tr 

echo "Rename Files.. "
mv normproto numhw.normproto 
mv inttemp numhw.inttemp 
mv pffmtable numhw.pffmtable 
mv shapetable numhw.shapetable  

echo "Create Tessdata.. "
combine_tessdata numhw. 
EOF
cp numhw.traineddata /Volumes/data/tesseract/share/tessdata/
cd ..
tesseract test.png output_1 -l numhw
#real:762408
#output:742408
pip install pytesseract 
#0.3.13
#clone源码
git clone git@github.com:zhongqiangwu960812/OpenCVLearning
mkdir doc-scan-ocr
cd doc-scan-ocr
cp -r ../OpenCVLearning/项目实战二_文档扫描ocr识别/images images
cp ../OpenCVLearning/项目实战二_文档扫描ocr识别/images/scan.jpg scan.jpg
jupyter nbconvert --to script ../OpenCVLearning/项目实战二_文档扫描ocr识别/jupyter/图像预处理.ipynb

pip install torch torchvision pretrainedmodels timm
mkdir parkinglot-glance
cd parkinglot-glance
mkdir cnn_pred_data
mkdir saved_model_weight
cp -r ../OpenCVLearning/项目实战三_停车场车位识别/test_images test_images
cp -r ../OpenCVLearning/项目实战三_停车场车位识别/train_data train_data
cp -r ../OpenCVLearning/项目实战三_停车场车位识别/video video

git clone git@github.com:YvanYan/image_processing
mkdir answer_sheet
cd answer_sheet
mkdir images
cp ../image_processing/answer_sheet/*.png images/


git clone git@github.com:ZhiqiangHo/Opencv-Computer-Vision-Practice-Python-
mkdir blob_from_images
cd blob_from_images
cp -r ../Opencv-Computer-Vision-Practice-Python-/"Chapter 18"/images images
cp ../Opencv-Computer-Vision-Practice-Python-/"Chapter 18"/bvlc_googlenet.caffemodel bvlc_googlenet.caffemodel
cp ../Opencv-Computer-Vision-Practice-Python-/"Chapter 18"/bvlc_googlenet.prototxt bvlc_googlenet.prototxt
cp ../Opencv-Computer-Vision-Practice-Python-/"Chapter 18"/synset_words.txt synset_words.txt

mkdir multi-object-tracking
cd multi-object-tracking
cp -r ../Opencv-Computer-Vision-Practice-Python-/"Chapter 19"/multi-object-tracking/videos videos

pip install dlib
mkdir multi-object-tracking-dlib
cd multi-object-tracking-dlib
cp -r ../Opencv-Computer-Vision-Practice-Python-/"Chapter 19"/multiobject-tracking-dlib/mobilenet_ssd mobilenet_ssd
cp ../Opencv-Computer-Vision-Practice-Python-/"Chapter 19"/multiobject-tracking-dlib/race.mp4 ./

pip install scikit-image
mkdir convolutions

pip install imutils
mkdir face-blink-detection
cd face-blink-detection
cp -r ../Opencv-Computer-Vision-Practice-Python-/"Chapter 21"/Face/images images
cp ../Opencv-Computer-Vision-Practice-Python-/"Chapter 21"/Face/shape_predictor_68_face_landmarks.dat ./
cp ../Opencv-Computer-Vision-Practice-Python-/"Chapter 21"/blink-detection/test.mp4 ./

pip install scikit-learn
wget -c https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
#huggingface-hub          0.25.2的版本，加载本地模型时也会报错
#ModuleNotFoundError: No module named 'huggingface_hub.utils._errors'
pip install huggingface-hub==0.24.5
git clone git@github.com:ultralytics/yolov5 yolov5-master
pip install yolov5
wget -c https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
git clone git@github.com:ultralytics/ultralytics  # clone
cd ultralytics
#pip install -e '.[dev]'
pip install '.[dev]'
wget -c https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
ln -s ../v8pt pt

conda create -n labelstudio -y python=3.10
conda activate labelstudio
#submit一组label以后转啊转的提示很久，打断检查，只有2个图片的labe被保存了
#有出错提示[tasks.models::set_lock::355] [ERROR] Current number of locks for task 2 is 1, but overlap=1: that's a bug because this task should not be taken in a label stream (task should be locked)
#3.7/1.9.2还是有这个问题，左边列表，右边打标，一张一张提交开始没问题，后来随机继续转啊转
#数据在.local里
rm -rf ~/.local/share/label-studio
pip install label-studio
label-studio
:<<EOF
mkdir label-studio-data
chmod 777 label-studio-data
rm -rf label-studio-data/*
nerdctl run -d -p 8080:8080 -v $(pwd)/catperson:/label-studio/catperson -v $(pwd)/label-studio-data:/label-studio/data --name label-studio heartexlabs/label-studio:latest
nerdctl logs -f label-studio
#docker一样有这个lock问题和转啊转的问题无法使用
EOF
#mac上用edge而不是safari打开，4张一起标还是有一样问题，左边列表，右边打标，一张一张提交没问题
#nginx没必要安装，预打标不是提供.pt模型文件的url，而是HumanSignal另外一个工具生成的服务程序跑起来以后提供的url
mkdir -p /data0/nginx/logs
nerdctl run -d --name nginx -p 9004:80 nginx
nerdctl cp nginx:/etc/nginx/conf.d/default.conf /data0/nginx/
nerdctl stop nginx && nerdctl rm nginx
#-v /workspace:/usr/share/nginx/html \
#-v /data0/nginx/conf:/etc/nginx \
#-v /workspace:/home/www \
:<<EOF
    location / {
        #root   /usr/share/nginx/html;
        #root   /home/www;
        root   /home/files;
        autoindex on;                            #开启目录浏览功能；   
        autoindex_exact_size off;            #关闭详细文件大小统计，让文件大小显示MB，GB单位，默认为b；   
        autoindex_localtime on;              #开启以服务器本地时区显示文件修改日期！   
        #index  index.html index.htm;
    }
EOF
nerdctl run -d -p 9004:80 --restart=always --name nginx \
-v /workspace:/home/files \
-v /data0/nginx/default.conf:/etc/nginx/conf.d/default.conf \
-v /data0/nginx/logs:/var/log/nginx \
nginx
#
conda create -n labelstudio-ml -y python=3.10
conda activate labelstudio-ml
git clone git@github.com:HumanSignal/label-studio
pip install .
#git clone git@github.com:HumanSignal/label-studio-sdk
#cd label-studio-sdk
#pip install .
cd ..
git clone git@github.com:HumanSignal/label-studio-ml-backend
#修改requirements.txt，删除对abel-studio-sdk的引用
cd label-studio-ml-backend
pip install .
#安装yolov8
label-studio-ml create yolov8_ml_backend
cp yolov8_ml_backend/model.py yolov8_ml_backend/model.py.bk
#修改加入Yolov8的模型逻辑
\cp /workspace/shouxiecv/search_on_2d/lsml-yolov8-model.py yolov8_ml_backend/model.py
export LABEL_STUDIO_URL=http://localhost:8080
export LABEL_STUDIO_API_KEY=d21316427326a59b28eb8342a339253b7c8024c2
export LABEL_STUDIO_MODEL_PATH=/workspace/shouxiecv/search_on_2d/v8pt/yolo11x.pt
export LABEL_STUDIO_MODEL_CONF=0.7
export LABEL_STUDIO_MODEL_VERSION=v11x
label-studio-ml start yolov8_ml_backend -p 9091
cd /workspace/search_on_2d
\cp ultralytics/ultralytics/cfg/datasets/coco.yaml catperson.yaml
#修改训练yaml数据文件
python yolov8-train.py
'''
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...
'''
wget -c https://ultralytics.com/assets/Arial.ttf -o /root/.config/Ultralytics/
'''
Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLO11n...
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt'...
'''
ln -s v8pt/yolo11n.pt yolo11n.pt


git clone git@github.com:frotms/PaddleOCR2Pytorch
cd PaddleOCR2Pytorch
cat << EOF > req.txt
shapely
numpy==1.23.5
pillow==9.5
pyclipper
opencv-python <= 4.2.0.32
torch==2.4.1
EOF
#2.5.0会报错ImportError: /data0/envs/paddleocr/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: symbol __nvJitLinkComplete_12_4, version libnvJitLink.so.12 not defined in file libnvJitLink.so.12 with link time reference
#和低版本cuda兼容有问题
pip install -r req.txt
pip install opencv-python opencv-contrib-python opencv-python-headless
pip install size scikit-image pyyaml torchvision==0.19.1
ln -s /workspace/models/cv/paddleocr/torch pth

#中文
python ./tools/infer/predict_det.py --image_dir ./doc/imgs/00009282.jpg --det_model_path pth/ch_ptocr_v4_det_server_infer.pth --det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml
python ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/ch/word_1.jpg --rec_model_path pth/ch_ptocr_v4_rec_server_infer.pth --rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml --rec_image_shape='3,48,320'

#多语言
python ./tools/infer/predict_det.py --det_algorithm DB --det_yaml_path ./configs/det/det_ppocr_v3.yml --det_model_path pth/en_ptocr_v3_det_infer.pth --image_dir ./doc/imgs/1.jpg
python ./tools/infer/predict_rec.py --image_dir ./doc/imgs_words/en/word_1.png --rec_model_path pth/en_ptocr_v4_rec_infer.pth --rec_yaml_path ./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml --rec_image_shape='3,48,320' --rec_char_dict_path ./pytorchocr/utils/en_dict.txt

cp -r doc/imgs doc/imgs.bk
mkdir doc/imgs-test-french && mv doc/imgs/french_*.jpg doc/imgs-test-french/
mkdir doc/imgs-test-german && mv doc/imgs/ger_*.jpg doc/imgs-test-german/
mkdir doc/imgs-test-japan && mv doc/imgs/japan_*.jpg doc/imgs-test-japan/
mkdir doc/imgs-test-korean && mv doc/imgs/korean_*.jpg doc/imgs-test-korean/
#串联使用方向分类
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path pth/ch_ptocr_mobile_v2.0_det_infer.pth --rec_model_path pth/ch_ptocr_mobile_v2.0_rec_infer.pth --use_angle_cls True --cls_model_path pth/ch_ptocr_mobile_v2.0_cls_infer.pth --vis_font_path ./doc/fonts/chinese_cht.ttf
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs-test-korean --det_model_path pth/ch_ptocr_mobile_v2.0_det_infer.pth --rec_model_path pth/ch_ptocr_mobile_v2.0_rec_infer.pth --use_angle_cls True --cls_model_path pth/ch_ptocr_mobile_v2.0_cls_infer.pth --vis_font_path ./doc/fonts/korean.ttf
#无法识别
#串联不使用方向分类
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_model_path pth/ch_ptocr_v4_det_server_infer.pth --det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml --rec_image_shape 3,48,320 --rec_model_path pth/ch_ptocr_v4_rec_server_infer.pth --rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml
python3 ./tools/infer/predict_system.py --image_dir ./doc/imgs --det_yaml_path ./configs/det/det_ppocr_v3.yml --det_model_path pth/en_ptocr_v3_det_infer.pth --rec_model_path pth/en_ptocr_v4_rec_infer.pth --rec_yaml_path ./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml --rec_image_shape='3,48,320'
#RuntimeError: Error(s) in loading state_dict for BaseModel:
python tools/infer/predict_e2e.py --e2e_model_path pth/en_server_pgnetA_infer.pth --image_dir ./doc/imgs_en/img623.jpg --e2e_algorithm PGNet --e2e_pgnet_polygon True --e2e_char_dict_path ./pytorchocr/utils/ic15_dict.txt --e2e_yaml_path ./configs/e2e/e2e_r50_vd_pg.yml
python tools/infer/predict_e2e.py --e2e_model_path pth/en_server_pgnetA_infer.pth --image_dir ./doc/imgs/1.jpg --e2e_algorithm PGNet --e2e_pgnet_polygon True --e2e_char_dict_path ./pytorchocr/utils/ppocr_keys_v1.txt --e2e_yaml_path ./configs/e2e/e2e_r50_vd_pg.yml
#不能识别
#超分辨率
python ./tools/infer/predict_sr.py --sr_yaml_path ./configs/sr/sr_telescope.yml --sr_model_path pth/sr_telescope_infer.pth --image_dir ./doc/imgs_words_en/word_52.png
:<<EOF
  File "/workspace/shouxiecv/PaddleOCR2Pytorch/tools/infer/pytorchocr_utility.py", line 150, in read_network_config_from_yaml
    if res['Architecture']['Head']['name'] == 'MultiHead' and char_num is not None:
KeyError: 'Head'
EOF
# 执行表格识别
mkdir ./inference_results/table

git clone git@github.com:PaddlePaddle/PaddleOCR.git
pip install -r requirements.txt
pip install paddlepaddle premailer openpyxl
ln -s /workspace/models/cv/paddleocr/pp tar
cd ppstructure
python3 predict_system.py --det_model_dir=../tar/ch_PP-OCRv4_det_server_infer \
                          --rec_model_dir=../tar/ch_PP-OCRv4_rec_server_infer \
                          --table_model_dir=../tar/ch_ppstructure_mobile_v2.0_SLANet_infer \
                          --layout_model_dir=../tar/picodet_lcnet_x1_0_fgd_layout_infer \
                          --image_dir=./docs/table/1.png \
                          --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
                          --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt \
                          --output=../output \
                          --vis_font_path=../doc/fonts/simfang.ttf
python3 predict_system.py --layout_model_dir=../tar/picodet_lcnet_x1_0_fgd_layout_infer \
                          --image_dir=./docs/table/1.png \
                          --output=../output \
                          --table=false \
                          --ocr=false \
                          --vis_font_path=../doc/fonts/simfang.ttf
python3 predict_system.py --det_model_dir=../tar/ch_PP-OCRv4_det_server_infer \
                          --rec_model_dir=../tar/ch_PP-OCRv4_rec_server_infer \
                          --table_model_dir=../tar/ch_ppstructure_mobile_v2.0_SLANet_infer \
                          --image_dir=./docs/table/table.jpg \
                          --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
                          --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt \
                          --output=../output \
                          --vis_font_path=../doc/fonts/simfang.ttf \
                          --layout=false
git clone git@github.com:datawhalechina/dive-into-cv-pytorch.git dive-into-cv-pytorch-orig
git clone git@github.com:PaddleOCR-Community/Dive-into-OCR Dive-into-OCR-orig




