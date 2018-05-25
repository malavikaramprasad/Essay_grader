pip install -U spacy
python -m spacy download en
pip install nltk
pip install pyenchant
pip install numpy
pip install scipy
pip install pandas
pip install futures
pip install --upgrade gensim
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install .
cd ..
chmod +x nltkdemo.py
python nltkdemo.py -test
