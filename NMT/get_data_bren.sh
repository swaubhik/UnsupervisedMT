# Swaubhik Chakraborty
# maharajbrahma

# Data config
N_MONO=87174 # NEED TO CHANGE HERE
CODES=6000
N_THREADS=48
N_EPOCHS=10 

SRC=en
TGT=gb # gb instead of brx for time being

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/$SRC-$TGT-data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para

# assuming tools path already exists
# create path
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl

INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fastBPE/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_RAW=$MONO_PATH/all.$SRC
TGT_RAW=$MONO_PATH/all.$TGT

SRC_TOK=$MONO_PATH/all.$SRC.tok
TGT_TOK=$MONO_PATH/all.$TGT.tok

BPE_CODES=$MONO_PATH/bpe_codes

CONCAT_BPE=$MONO_PATH/all.$SRC-$TGT.$CODES
SRC_VOCAB=$MONO_PATH/vocab.$SRC.$CODES
TGT_VOCAB=$MONO_PATH/vocab.$TGT.$CODES
FULL_VOCAB=$MONO_PATH/vocab.$SRC-$TGT.$CODES

SRC_VALID=$PARA_PATH/dev/dev.$SRC
TGT_VALID=$PARA_PATH/dev/dev.$TGT

SRC_TEST=$PARA_PATH/dev/test.$SRC
TGT_TEST=$PARA_PATH/dev/test.$TGT


# Download Monolingual data
cd $MONO_PATH

echo "Downloading English files"

# file size is too big 37G
#wget -c -O "all.en" "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/en.txt"
head -174374 /home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/en-brx/en.txt > all.$SRC

echo "Downloading Bodo files"

wget -c -O "all.$TGT" "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/bd.txt"

# --------------------------------------
# remove empty lines and change the N_MONO
# remove empty lines from SRC_RAW and TGT_RAW and save it in the same file
sed -i '/^$/d' $SRC_RAW
sed -i '/^$/d' $TGT_RAW

#head $SRC_RAW
#head $TGT_RAW

# make both the files of same length
head -n $N_MONO $SRC_RAW > $SRC_RAW.tmp
head -n $N_MONO $TGT_RAW > $TGT_RAW.tmp
mv $SRC_RAW.tmp $SRC_RAW
mv $TGT_RAW.tmp $TGT_RAW
# ----------------------------------------------

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your EN monolingual data."; exit; fi
if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your BRX monolingual data."; exit; fi

# tokenize data with Moses [Better to use those tokenizer for BRX]
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TOK
fi

echo "EN monolingual data tokenized in: $SRC_TOK"
echo "BRX monolingual data tokenized in: $TGT_TOK"

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TOK.$CODES" && -f "$TGT_TOK.$CODES" ]]; then
  echo "Applying BPE codes..."
  $FASTBPE applybpe $SRC_TOK.$CODES $SRC_TOK $BPE_CODES
  $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES
fi
echo "BPE codes applied to EN in: $SRC_TOK.$CODES"
echo "BPE codes applied to BRX in: $TGT_TOK.$CODES"

# extract vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TOK.$CODES > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TOK.$CODES > $TGT_VOCAB
  $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
fi
echo "EN vocab in: $SRC_VOCAB"
echo "BRX vocab in: $TGT_VOCAB"
echo "Full vocab in: $FULL_VOCAB"


# binarize data
if ! [[ -f "$SRC_TOK.$CODES.pth" && -f "$TGT_TOK.$CODES.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TOK.$CODES
  $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
fi
echo "EN binarized data in: $SRC_TOK.$CODES.pth"
echo "BRX binarized data in: $TGT_TOK.$CODES.pth"

# Get dev and test set

cd $PARA_PATH

echo "Downloading dev set"
wget -c https://indictrans2-public.objectstore.e2enetworks.net/flores-22_dev.zip

echo "Downloading test set"
wget -c https://indictrans2-public.objectstore.e2enetworks.net/IN22_testset.zip


echo "Extracting dev parallel data..."
unzip flores-22_dev.zip

echo "Extracting test parallel data..."
unzip IN22_testset.zip

# make ./dev directory
mkdir dev

# Rename file name

mv $PARA_PATH/flores-22_dev/all/eng_Latn-brx_Deva/dev.eng_Latn $SRC_VALID
mv $PARA_PATH/flores-22_dev/all/eng_Latn-brx_Deva/dev.brx_Deva $TGT_VALID

mv $PARA_PATH/IN22_testset/gen/test.eng_Latn $SRC_TEST
mv $PARA_PATH/IN22_testset/gen/test.brx_Deva $TGT_TEST

echo "Tokenizing valid and test data..."
cat $SRC_VALID | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID.tok
cat $TGT_VALID | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_VALID.tok

cat $SRC_TEST | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST.tok
cat $TGT_TEST | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TEST.tok

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_VALID.tok.$CODES $SRC_VALID.tok $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_VALID.tok.$CODES $TGT_VALID.tok $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_TEST.tok.$CODES $SRC_TEST.tok $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_TEST.tok.$CODES $TGT_TEST.tok $BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
# rm -f $SRC_VALID.tok.$CODES.pth $TGT_VALID.tok.$CODES.pth $SRC_TEST.tok.$CODES.pth $TGT_TEST.tok.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID.tok.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID.tok.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.tok.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST.tok.$CODES

#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    EN: $SRC_TOK.$CODES.pth"
echo "    FR: $TGT_TOK.$CODES.pth"
echo "Parallel validation data:"
echo "    EN: $SRC_VALID.tok.$CODES.pth"
echo "    FR: $TGT_VALID.tok.$CODES.pth"
echo "Parallel test data:"
echo "    EN: $SRC_TEST.tok.$CODES.pth"
echo "    FR: $TGT_TEST.tok.$CODES.pth"
echo ""

#
# Train fastText on concatenated embeddings
#

# if ! [[ -f "$CONCAT_BPE" ]]; then
#   echo "Concatenating source and target monolingual data..."
#   cat $SRC_TOK.$CODES $TGT_TOK.$CODES | shuf > $CONCAT_BPE
# fi
# echo "Concatenated data in: $CONCAT_BPE"

# if ! [[ -f "$CONCAT_BPE.vec" ]]; then
#   echo "Training fastText on $CONCAT_BPE..."
#   $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE
# fi
# echo "Cross-lingual embeddings in: $CONCAT_BPE.vec"


#* Train monolingual embeddings separately for each Language, and align them with MUSE
# Train fastText on source language
if ! [[ -f "$SRC_TOK.$CODES.vec" ]]; then
    echo "Training fastText on $SRC_TOK.$CODES..."
    $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $SRC_TOK.$CODES -output $SRC_TOK.$CODES
fi
# Train fastText on target language
if ! [[ -f "$TGT_TOK.$CODES.vec" ]]; then
    echo "Training fastText on $TGT_TOK.$CODES..."
    $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $TGT_TOK.$CODES -output $TGT_TOK.$CODES
fi
echo "Monolingual embeddings in: $SRC_TOK.$CODES.vec and $TGT_TOK.$CODES.vec"

# Align monolingual embeddings using MUSE
if ! [[ -f "$SRC_TOK.$CODES.vec.align" && -f "$TGT_TOK.$CODES.vec.align" ]]; then
    echo "Aligning monolingual embeddings using MUSE..."
   CUDA_VISIBLE_DEVICES=1,2 python $TOOLS_PATH/MUSE/unsupervised.py --src_lang $SRC --tgt_lang $TGT --src_emb $SRC_TOK.$CODES.vec --tgt_emb $TGT_TOK.$CODES.vec --emb_dim 512 --dis_most_frequent 4000 
fi

# for seperately run MUSE
# CUDA_VISIBLE_DEVICES=1,2 python tools/MUSE/unsupervised.py --src_lang en --tgt_lang gb --src_emb /home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/NMT/en-gb-data/mono/all.en.tok.6000.vec --tgt_emb /home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/NMT/en-gb-data/mono/all.gb.tok.6000.vec --emb_dim 512 --dis_most_frequent 4000 