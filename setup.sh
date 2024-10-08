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
