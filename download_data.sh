DATA="za-traffic-2020"
WORK_DIR="/content"
pip install kaggle --upgrade
echo "{\"username\":\"ptran1203\",\"key\":\"8984c9ec69a9482990fafc7a8d60015a\"}" > kaggle.json
if [ ! -d ~/.kaggle ]; then
    mkdir ~/.kaggle
fi
if [ ! -d $WORK_DIR/dataset ]; then
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    kaggle datasets download phhasian0710/$DATA
    unzip $DATA.zip -d $WORK_DIR/dataset
    rm $DATA.zip
fi
