# cnn-sentiment-classification
Convolutional neural networks for sentiment classification.
## Requirement
Python: 3.5.1  
Chainer: 1.19.0  
Stanford Sentiment Treebank (SST)をdatasetとして使っています。[こちら](https://nlp.stanford.edu/sentiment/)からダウンロードできます。  
単語分散表現はword2vecの学習済みモデル(`GoogleNews-vectors-negative300.bin`)を使ってます。  
[こちら](https://code.google.com/archive/p/word2vec/)でダウンロードできます。  

## Usage
1. `GoogleNews-vectors-negative300.bin`をダウンロードし、`vector`ファイルに入れてください。
2. SST dataset(`train.txt`,`dev.txt`,`test.txt`)をダウンロードし、`data`ファイルに入れてください。
3. `pip install -r requirement.txt`を実行してください。
4. `create_label_sentence.py`を実行して、訓練データ、開発データ、テストデータを作ってください。第一引数には[train,dev,test]、第二引数には[2,5]が入ります。  
例えば、SSTの2値分類(positive or negative)の訓練データを作りたい場合は以下の通りです。  

```
python create_label_sentence.py train 2
```
4. 実行  

```
python cnn_max.py --gpu 1 --traindata ./data/sst2_train_label_sentence.txt --devdata ./data/sst2_dev_label_sentence.txt --testdata ./data/sst2_test_label_sentence.txt --classtype 2
```

## Optional arguments
    --gpu GPU                      use gpu number -1: use cpu
    --traindata TRAINDATA          default: SST5
    --devdata DEVDATA              default: SST5
    --testdata TESTDATA            defalut: SST5
    --batchsize BATCHSIZE          default: 50
    --epoch EPOCH                  default: 25  
    --classtype CLASSTYPE          default: 5   

## Data format
正解ラベルと単語列の間に空白を入れてください。  
例(very positiveの文)

```
4 A very funny movie .
```
