 
## Generative Models with an Application for Professional Tennis Data


The aim of the project is to utilise modern Language type models on tennis shot data, which due to it's sequential (non IID) structure over a series of shots, 
shows similar structure to language data. Hence many of the same methods such as RNN's and transformers can be considered, in particular to generate sythetic points data to inform strategy 
and to extract point embeddings that could be used in downstream applications. For more details see the presentation Tennis_shot_representations.pptx

Full credit for the data from the incredible resource to Jeff Sackmann

[JeffSackmann](https://github.com/JeffSackmann/tennis_MatchChartingProject)


* Code structure


1. Clone the repo or download the files
2. use conda to setup an environment with the requirements.txt file
3. activate the new environment
4. chdir to the root directory of tennis_gpt.py
5. use the code as per below


Lets see how it works

* All code is in Python/Pytorch but there are many ways to do it –
Keras, Tensorflow, Hugginface, FASTAI, OPENAI etc depending
upon whether you want to work with low level code or a higher
level API. The code is adapted from nanoGPT
https://github.com/karpathy/nanoGPT
* Run python tennis_gpt.py -h to access help
* To train **python tennis_gpt.py -i tennis_shots_new_all_final_reduced.txt -o tennis_gpt --type transformer --max-steps 10000**
* To sample from a trained model **python makemore_tennis_clean.py -i tennis_shots_new_all_final_reduced.txt -it <initial_token list e.g. a114,f39> -o tennis_gpt --type transformer --sample-only**
* To generate point embeddings use **embedding_representations.py**



What does the code do?

1\. Generate new points data with and without prompts. This is the impressive part – think of ChatGPT. Prompting is the upside of this sequential sturcture. This code also plots the individual shot embeddings as well.

2\. To extract point embeddings – like document embeddings so we can cluster and extract similar points by semantic search. It may
ultimately help us uncover latent strategies The Generation part is what is getting all of the attention now,
particularly with the emergent capabilities of these Large
Language Models – simply doing things no one really expected.
However long term the ability of these models to uncover latent
structure via the embeddings may prove just as useful.

Going to need Data and lots of it!!

* Unfortunately the Hawkeye Data is not publicly available - yet

* However an amazing project at github.com/JeffSackman runs a match charting project

* Each point is coded as a sequence of shots

\- shot type,direction,<depth>,<outcome>

\- 1 f/h side, 2 middle, 3 b/h side

\- f1 -> forehand hit to forehand side

\- b3\* -> backhand cross court winner

* A point is a series of shots, with a beginning (serves a?? and an ending ??\*/@/#

a214,b28,f1\*

a116,b29,f1,f1,b2,b2n@
    
for a more detailed description of each point see


[Point Definitions])https://github.com/JeffSackmann/tennis_MatchChartingProject/blob/master/MatchChart%200.3.2.xlsm)

* The important point is that there is some sequential structure over a series of shots

How is the data prepared??
    
![Image](Data_Prep.png)


Transformer Architecture
    
![Image](Transformer.png)



<a name="br13"></a> 

