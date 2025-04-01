Required Pre-processing steps for each dataset:

#1 GTN
Ready to go.

#2 SwitchBoard
Ready to go.

#3 CHiME-6
Ready to go.

#4 AMI
Generate additional examples:
For each "file" field in the json:
    Randomly combine segments if segment_end == segment_start to get new examples
    Note: Must be done for both normalized and unnormalized fields

- I want a script to combine these sentences to generate new sentences at random. 
We can decide the number of sentences we would generate later, when it's time to train the model.
- accept a multiplier? So if someone passes, say 0.5, then you generate 0.5 as many 
combinations as there are segments for that file.

#5 EarningsCall
Data as lists of words => Randomly combine consecutive words to form sentences for each id.
Randomly select normalized form based on probability
Correct normalized form in cases of "hundred"/"thousand"/"million"/"billion", etc.

select the normalized form based on probability, but some of the lower 
probability versions are also good to have as training data for the model. 
Maybe you can distribute it as: pick the most likely version 60% of the times, 
pick the second most likely 30% of the times, and pick one of the rest at 
random 10% of the time. Again, I want to have a script that can do this at 
random, and the exact number to generate we will decide later on. 

If you look at the way the words "thousand"/"million"/"billion" appear in 
the dataset, you'll see the normalized forms are wrong. For example:
For "$25 million", it's normalized form appears as "twenty five dollars 
million", which is wrong. So wherever these words appear, you would have to 
fix the normalized form so that it is "twenty five million dollars".


#6 SPGISpeech
Ensure unnormalized and normalized words match up 100% 
(there may be speech recognition errors)
Use Sequence Matching to replace error words with correct words
Filter sentences that can't be corrected
Process abbreviations in unnormalized form ("USA" to "u s a") 
and convert sentences to lower case

filter means delete.
 
To get abbreviations, I usually use a regex to match uppercase words 
(may be separated by .) like USA, U.S.A, or U.S.A.
 
The whole sentence is to be converted to lower case.

But you would have to remove all punctuations (except apostrophe (')) 
from the unnormalized sentence before you perform matching. 
Also note that we don't want numbers and symbols to end up in the 
normalized form, so if the added/deleted/substituted part contains 
numbers and symbols, then we don't want to perform replacement.       