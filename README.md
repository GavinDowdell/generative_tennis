<a name="br1"></a> 

Generative Models with an

Application for Professional Tennis

Data

Dr Gavin Dowdell



<a name="br2"></a> 

Why Machine Learning is great area

• Always new things to learn

• Always new areas of application

• I was asked to talk about a technical area of interest so I

thought this area may interest the group.

• Outline

1\. Quickly describe the Language Modelling problem via

Generative Models

2\. Show an application from a personal project of mine. Personal

projects are great opportunities to learn and good fun

Maybe it will generate some ideas for your own projects!!



<a name="br3"></a> 

What are generative models

• Generative Models are models that generate synthetic probable

realisations

• The models to do so are still essentially Machine Learning

models except the target variable is interpreted as a parameter of

a probability distribution and can therefore be sampled for

realisations.

• For discrete distributions such as language commonly p(t;**θ**) =

Cat(**θ**)

• E[**θ**]= f(**x**;**ω**) via a particular ML model and then sample from p()

Good to remember we are sampling from probability distributions

even though the applications are incredibly good and look like

magic!

• The challenging part isn’t really to generate the realisation, as



<a name="br4"></a> 

Generative Models are not new!!

Many years ago I was employed to price Energy options which involved

generating many synthetic price distributions – over a continuous

distribution.

Pricing Options. Price approx = the discounted expected payoff for

strike K.

Asymmetric Payoff

Max(S<sub>t</sub> – K,0)

Need to generate

a large number of

possible price

paths

Strike =

14



<a name="br5"></a> 

Generative Models are not new but things are

more difficult with discrete sequential data

• For large discrete distributions such as Natural Text, where the distribution is

now categorical over the size of the token Vocab (1000’s), then this becomes

more difficult.

• As if things weren’t hard enough, if the data is sequential such that values are

dependent upon the previous values, hence non IID or autoregressive, then this

becomes much, much more difficult unless you make assumptions such as

p(y |**y** ) = p(y |y ,y … y ) = p(y |y ,y ) - trigrams -> limited context

n **n-1**

n n-1 n-2

1

n n-1 n-2

• Language is not IID – a word distribution is based upon the previous words

“Someone making others do Maths on a Friday afternoon is ????”

• The sampling distribution for ???? Is dependent on the whole sentence/prompt



<a name="br6"></a> 

Everyone is familiar now with the success

around language models

• Language tends to have long-term

dependencies as words really only have

their meaning derived from a longer

context

• A few pretraining options but a self-

supervised Language Model (aka predict

the next word model) is a popular

pretraining architecture. Don’t have to

worry about collecting targets for a self-

supervised model – game changer.

• Generative Pretrained Models (including

the GPT family), which implement

Language Models, have been particularly

successful because of their architecture

and the vast quantity of data they were

trained on.



<a name="br7"></a> 

However what about autoregressive

sequential data that has some similarities

with language?

• Music notation is an example that is

structured like language

• The current note is not independent

of the previous notes

• I am not musical but I like sports so

I was interested in understanding

it’s sequential structure

• Hence many NLP techniques and

architectures can be applied,

although not usually pretrained

models which are specific to

language



<a name="br8"></a> 

Shots within a tennis point as like words

in a sentence

• Professional Tennis players are like chess players – setting up

strategies many shots ahead.

• Unlike us their play is not random – Rafa Nadal a great

example.

• Play Video

• It ties into a core concept in Machine Learning – observable vs

latent. What we see during a sporting event looks quite

unstructured however it is more likely to be a noisy

representation of some much simpler latent behaviour.

• Can we discover that latent behaviour with a language model?



<a name="br9"></a> 

What are we actually trying to achieve

here?

1\. Generate new points data with and without prompts. This is the

impressive part – think of ChatGPT. Prompting is the upside of this

sequential sturcture

2\. To extract point embeddings – like document embeddings so we

can cluster and extract similar points by semantic search. It may

ultimately help us uncover latent strategies

The Generation part is what is getting all of the attention now,

particularly with the emergent capabilities of these Super Large

Language Models – simply doing things no one really expected.

However long term the ability of these models to uncover latent

structure via the embeddings may prove just as useful.



<a name="br10"></a> 

Going to need Data and lots of it!!

• Unfortunately the Hawkeye Data is not publicly available - yet

• However an amazing project at github.com/JeffSackman runs a match charting

project

• Each point is coded as a sequence of shots

\- shot type,direction,<depth>,<outcome>

\- 1 f/h side, 2 middle, 3 b/h side

\- f1 -> forehand hit to forehand side

\- b3\* -> backhand cross court winner

• A point is a series of shots, with a beginning (serves a?? and an ending ??\*/@/#

a214,b28,f1\*

a116,b29,f1,f1,b2,b2n@

• The important point is that there is some sequential structure over a series of shots



<a name="br11"></a> 

How is the data prepared??

1\.

extracted from the dataset. In this case 257

discrete values in the Vocab

Assume the data vocab has been

Target

seq

2\.

Use the Vocab to numericalise the data

and add <bos> and <eos> tokens to understand

Input

seq

start and

finish

3\.

The key is how to prepare the training

dataset

Could build the training dataset this way

0 ꢀ22

0,22 ꢀ54

0,22,54 ꢀ5

4\.

But seq2seq model is more efficient.

During training use autoregressive teacher

forcing which means that the correct answer is

fed back in even if it is not the predicted value.

During inference feed the predicted value in as the

next value.



<a name="br12"></a> 

The transformer architecture - briefly

• Transformers within the recent language

models so quite famous now.

• “Attention is all you need” – Vaswani et.

al. adapted to language models

• A single block does a few things but well

known for self attention which produces

layers of contextualised representations

by attending to different parts of the input

with the target in mind.

• Parallel computation and long-range

dependencies subject to context length

• The goal is simple to produce better

representations to predict the target.

• 1. Better prediction = better generative

model or

2\. maybe we just want the

representation?



<a name="br13"></a> 

Lets see how it works

• All code is in Python/Pytorch but there are many ways to do it –

Keras, Tensorflow, Hugginface, FASTAI, OPENAI etc depending

upon whether you want to work with low level code or a higher

level API. The code is adapted from nanoGPT

https://github.com/karpathy/nanoGPT

• What can go wrong with a real time demo ha?

Run demo



<a name="br14"></a> 

In Conclusion

• Hopefully this example demonstrates how to apply the learning from

Generative Language Models to other areas that have sequential

discrete data. Of course it is still mostly used for NLP.

• Other examples include

1\. Medical Codes. Medical patients/injured workers tend to receive a

time indexed set of medical treatments identified by discrete

treatment codes which form a language like sequence –

embeddings and semantic search could be useful to identify

effective treatment patterns?

2\. Biology – a lot of biological structures are discrete and sequential.

Some people think it will revolutionise the area.

3\. Anything you can think of, if you have a good idea give it a try!!

