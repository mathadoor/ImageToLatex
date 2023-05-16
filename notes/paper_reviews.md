This documents serves to note down my reviews on papers I read for this project. 

### 1. [CROHME 2011 Competition on Recognition of Online Handwritten Mathematical Expressions](../data/ICFHR_package/CROHME_papers/CROHME_ICDAR_2011.pdf)
The training set consists of 921 expression and supplies the underlyin ggrammar for understanding the content of the
training data. The test data consists of 348 expressions. The evalutation criteria is based on four different aspects of
the recognition problem. At the time of publication, the best expression level recognition accuracy is 22.41%.

The handwritten mathematical expressions are saved in InkML format. Two parts are defined in the training dataset containing 
296 and 921 expressions respectively. The second part includes the expressions from the 
first part. Part-I expressions are less complex. The expressions are accompanied by their underlying grammar. The test expressions
are different from teh training set. The test set is also divided into two parts just like the training set. The first part contains
181 expressions and the second part contains 348. 

What does an InkML file contain?  
The InkML contains three kinds of information:
1. The trace information: The trace information is the actual ink data. It contains the x and y coordinates of the pen tip.
2. The annotation information: it contains the segmentation and label information of each symbol of the expression.
3. The expression level ground truth as MathML structure. Here is an example of an InkML file for the expression a < b / c:

```InkML
<ink xmlns="http://www.w3.org/2003/InkML">
<traceFormat>
    <channel name="X" type="decimal"/>
    <channel name="Y" type="decimal"/>
</traceFormat>
<annotation type="writer">w123</annotation>
<annotation type="truth">$a<\frac{b}{c}$</annotation>
<annotation type="UI"> 2011_IVC_DEPT_F01_E01 </annotation>
<annotationXML type="truth" encoding = "Content-MathML">
    <math xmlns="http://www.w3.org/1998/Math/MathML">
        <mrow>
            <mi xml:id="A">a</mi>
            <mrow>
                <mo xml:id="B"><</mo>
                <mfrac xml:id="C">
                    <mi xml:id="D">b</mi>
                    <mi xml:id="E">c</mi>
                </mfrac>
            </mrow>
        </mrow>
    </math>
</annotationXML>
<trace id="1">985 3317, ..., 1019 3340</trace>
...
<trace id="6">1123 3308, ..., 1127 3365</trace>
    <traceGroup xml:id="7">
        <annotation type="truth">Ground truth</annotation>
        <traceGroup xml:id="8">
            <annotation type="truth">a</annotation>
            <annotationXML href="A"/>
                <traceView traceDataRef="1"/>
            <traceView traceDataRef="2"/>
        </traceGroup>
        ...
    </traceGroup>
</ink>
```

### 2. [CROHME 2012 Competition on Recognition of Online Handwritten Mathematical Expressions](../data/ICFHR_package/CROHME_papers/CROHME_ICFHR_2012.pdf)

### 3. [Image to Latex Dataset - Kaggle](https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k)

### 4. [Named Entity Recognition with Bidirectional LSTM-CNNs](https://aclanthology.org/Q16-1026.pdf)

[2011-2014 Dataset](https://www.kaggle.com/datasets/rtatman/handwritten-mathematical-expressions)

[2023 CROHME DATASET](https://crohme2023.ltu-ai.dev/data-tools/)  
It seems that the dataset consists of the dataset from the previous cases. We have essentially
three different type of challenges in this:  
1. Online Handwritten Task: in this case the input comes as polylines from a pen. The algorithm is supposed to 
recognize the expression, the bounding box of each symbol and the label of each symbol.
2. Offline Handwritten Task: in this case, the input is a set of prewritten equations.
3. BiModal Handwritten Task: In this case, we have both the poly lines and the prewritten equations as the input.
The dataset is also much bigger than the previous ones. It has over 10979 inkml files with 150k aritifically generated
files. We also have 1045 real equations. The validation set is based on CROHME2016_test set of size 1147. 
