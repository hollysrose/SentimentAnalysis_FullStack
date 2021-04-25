package edu.ucmo.spring_bare_bones;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

public class Vectorizer {

    ArrayList<String> stopwords = new ArrayList<>();

    HashSet<String> dictionary = new HashSet<>();
    ArrayList<ArrayList<String>> trainingReviews = new ArrayList<>();

    ArrayList<String> trainingLabels = new ArrayList<>();
    ArrayList<double[]> trainingAnswers = new ArrayList<>();

    ArrayList<String[]> tokens = new ArrayList<>();
    ArrayList<String> organizedTokens = new ArrayList<>();


    ArrayList<String> indexedDictionary = new ArrayList<>();
    int frequencyThreshold = 50;
    int thresholdCrossingTokens = 0;

    ArrayList<String> newDictionary = new ArrayList<>();
    ArrayList<String> finalDictionary = new ArrayList<>();

    double[] inputBiases;

    ArrayList<ArrayList<String>> testReviews = new ArrayList<>();

    ArrayList<String> testLabels = new ArrayList<>();
    ArrayList<double[]> testAnswers = new ArrayList<>();

    ArrayList<double[]> trainingMatrix = new ArrayList<>();
    ArrayList<double[]> testMatrix = new ArrayList<>();
    /*
    train() reads reviews in, cleans the text, reduces to features, creates a
    "bag of words" (dictionary of unique words in our training dataset), and forms an input matrix of vectors.
    */
    public void readyTrainingData(String trainingDataLocation) throws IOException {
        //-------------------------------------------------------------------------------------------------------------
        /*
        There are two parts of every review in our training dataset:
            Part 1) words of the review (which are located between " ", (comma included)
            Part 2) sentiment label (either "neg" or "pos").
        Every review is:
            " [some line(s) of text until] ",neg OR ",pos
        Example:
            "Wow, gee! This was a great example, Batman!",pos
        */
        //-------------------------------------------------------------------------------------------------------------

        //In the code block below are four key objects that are referenced/built by all methods that follow.
        //-------------------------------------------------------------------------------------------------------------
        /*
        Creating File Object to hold our training dataset that is in .csv format.
        Passing to BufferedReader so that readLine() method is available, rather than reading character at a time.
        */
        File csv = new File(trainingDataLocation);
        FileReader fileReader = new FileReader(csv);
        //KEY OBJECT 1.)
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        /*
        ArrayList<String> stopwords created here for method cleanText() to refer to so that ArrayList does not need
        to be formed as each line is cleaned.
        */
        //KEY OBJECT 2.)


        /*
        Three ArrayLists.
        Two will have a name, and so will be outside of the for-loop below. One is in the for-loop below.
        Unnamed (meaning that it is within for-loop, is passive):
            ArrayList<String> trainingReview  stores lines of a review as they are read in.
        Named (outside of for-loop, meaning it will be called later):
            ArrayList<ArrayList<String>> trainingReviews. Stores each review (each ArrayList<String> trainingReview of the
            for-loop).
            ArrayList<String> trainingLabels. Stores pos or neg label of each training review, in order.
        */
        //KEY OBJECTS 3.) and 4.)

        //-------------------------------------------------------------------------------------------------------------
        gatherStopwords();
        for(String line; (line = bufferedReader.readLine()) != null; ) {
            //Remember: ArrayList<String> trainingReview to create review by storing lines of review
            ArrayList<String> trainingReview = new ArrayList<>();

            /*
            We need to find where a review ends. (Find sentiment label at end of review.)
            */
            /*
            char[] last3 that will store last 3 characters of line as we search for match to either [n][e][g]
            or [p][o][s] at the end of a line.
            */
            char[] last3 = new char[3];
            //Labels we are looking for
            char[] pos = {'p','o','s'};
            char[] neg = {'n','e','g'};
            //Go to end of line, grab last three characters, throw into char[] last3
            char[] lineCharArray = line.toCharArray();
            for(int i = 2; i >=0 ; i--)
                last3[2 - i] = lineCharArray[lineCharArray.length - (i + 1)];
            /*
            If last3 does not equal either pos or neg label, add line to current review
            (as element to our review, ArrayList<String> trainingReview)
            */
            boolean labelFound = false;
            if(Arrays.equals(last3,pos) || Arrays.equals(last3,neg))
                labelFound = true;
            if(labelFound == false)
                trainingReview.add(cleanText(line, stopwords));
            /*
            If last3 does equal either pos or neg label, then need both training label and
            line with training label cut off.
            */
            else{
                //Cast last3 ("pos" or "neg") into a String (the pos or neg label)
                String label = new String(last3);
                //Add label to ArrayList<String> trainingLabels
                trainingLabels.add(label);
                //Snip training label off end of line
                String lineWithoutLabel = line.substring(0, line.length() - 3);
                //Add line to review
                trainingReview.add(cleanText(lineWithoutLabel, stopwords));
            }

            //Add review to clean dataset of training reviews
            trainingReviews.add(trainingReview);

        }

        //Create correct answer vector for neural network to understand. 1 = positive, 0 = negative.
        for(int i = 0; i < trainingLabels.size(); i++){
            double[] answer = new double[2];
            if(trainingLabels.get(i).equals("pos")){
                answer[0] = 1;
                answer[1] = 0;
                trainingAnswers.add(answer);
            }
            else{
                answer[0] = 0;
                answer[1] = 1;
                trainingAnswers.add(answer);
            }

        }

        findFeatures();

        //Fill the trainingMatrix that will feed the neural network for training
        for(int i = 0; i < trainingReviews.size(); i++){
            double[] vector;
            vector = vectorize(trainingReviews.get(i));
            trainingMatrix.add(vector);
        }

        /*
        //Checking on trainingMatrix
        int vectorCount = 0;
        for(int i = 0; i < trainingMatrix.size(); i++){
            for(int j = 0; j < trainingMatrix.get(i).length; j++){
                System.out.print(trainingMatrix.get(i)[j]);
            }
            vectorCount++;
        }
        System.out.println("InputMatrix vectors: " + vectorCount);
        */
        //-------------------------------------------------------------------------------------------------------------
    }

    public void gatherStopwords() throws IOException {
        File stopwordsDoc = new File("src/main/resources/stopwords.txt");
        FileReader stopwordsReader = new FileReader(stopwordsDoc);
        BufferedReader bufferedStopwordsReader = new BufferedReader(stopwordsReader);
        for(String stopword; (stopword = bufferedStopwordsReader.readLine()) != null; ){
            stopwords.add(stopword);
        }
    }

    public String cleanText(String line, ArrayList<String> stopwords) throws IOException {
        //Remove non-word and non-sentiment characters
        String useless = "[\u0097\u0096®§¢¡!#$%^&*¿?,.;?\"“0-9/;():_~`‘£₤½–\uF0B7¾¨»«·+-]";
        line = line.replaceAll(useless,"");
        line = line.replaceAll("'","");
        //Lowercase
        line = line.toLowerCase();
        /*
        We want to keep negative sentiment meanings, while removing non-sentimental 'could', 'did', etc.
        We can also simplify from myriad negatives ('couldn't, didn't) to a consistent form of negative ('no').
        Replace all ***n't (such as 'couldn't', 'didn't') with 'no'.
        Remember that we already removed the '.
        */
        String shouldnt = "shouldnt";
        String dont = "dont";
        String arent = "arent";
        String couldnt = "couldnt";
        String didnt = "didnt";
        String doesnt = "doesnt";
        String hadnt = "hadnt";
        String hasnt = "hasnt";
        String havent = "havent";
        String isnt = "isnt";
        String mightnt = "mightnt";
        String mustnt = "mustnt";
        String neednt = "neednt";
        String shant = "shant";
        String wasnt = "wasnt";
        String werent = "werent";
        String wont = "wont";
        String wouldnt = "wouldnt";
        String cant = "cant";

        line = line.replaceAll(shouldnt, "no").replaceAll(dont, "no").replaceAll(arent, "no")
                .replaceAll(couldnt, "no").replaceAll(didnt, "no").replaceAll(doesnt, "no")
                .replaceAll(hadnt, "no").replaceAll(hasnt, "no").replaceAll(havent, "no")
                .replaceAll(isnt, "no").replaceAll(mightnt, "no").replaceAll(mustnt, "no")
                .replaceAll(neednt, "no").replaceAll(shant, "no").replaceAll(wasnt, "no")
                .replaceAll(werent, "no").replaceAll(wont, "no").replaceAll(wouldnt, "no")
                .replaceAll(cant,"no");

        /*
        Create unique "anti-words" that are associated with negative reviews.
        Example: 'no love' becomes '-love', which is different than 'love'.
        */
        line = line.replaceAll(" " + "no" + " ", " " + "-");

        //Remove indicator of line breaks
        line = line.replaceAll("<br ><br >", " ");

        //Remove stopwords
        for(int i = 0; i < 81; i++)
            line = line.replaceAll(stopwords.get(i) + " ", "");
        /*
        We also remove stopwords here. However, this step only removes instances of the stopword
        that are surround by a space so that we do not remove all 'on' from 'diction_(space)_'.
        */
        for(int i = 75; i < stopwords.size(); i++)
            line = line.replaceAll(" " + stopwords.get(i) + " ", " ");

        return line;
    }

    public void findFeatures(){
        //Tokenizing
        //-------------------------------------------------------------------------------------------------------------
        //Breaking trainingReviews into individual tokens (NLP term for "words", essentially)

        for(int i = 0; i < trainingReviews.size(); i++){
            for(int j = 0; j < trainingReviews.get(i).size(); j++) {
                String[] tokensGathered = trainingReviews.get(i).get(j).trim().split("\\s+");
                tokens.add(tokensGathered);
            }
        }

        //Creating an alphabetical order of tokens, collected into a single object

        for(int i = 0; i < tokens.size(); i++){
            for(int j = 0; j < tokens.get(i).length; j++){
                organizedTokens.add(tokens.get(i)[j]);
            }
        }
        Collections.sort(organizedTokens);

        /*
        From our tokens, making piles of words to make a dictionary.
        In other words, this is the dictionary of all the words we've encountered in training.
        */
        for(int i = 0; i < tokens.size(); i++){
            for(int j = 0; j < tokens.get(i).length; j++) {
                dictionary.add(tokens.get(i)[j]);
            }
        }

        /*
        How often does a dictionary word appear in our training data?
        int[] frequency that's same size as dictionary and initialized to zero.
        As we count a token as an instance of a dictionary word, the corresponding index
        in int[] frequency will increase.
        */
        double[] frequency = new double[dictionary.size()];
        for(int i = 0; i < frequency.length; i++){
            frequency[i] = 0;
        }
        //Since HashSet doesn't have an index, and we will need to iterate through the dictionary...
        Object[] indexedDictionaryArray = dictionary.toArray();
        for(int i = 0; i < indexedDictionaryArray.length; i++){
            indexedDictionary.add((String) indexedDictionaryArray[i]);
        }
        Collections.sort(indexedDictionary);
        //Filling out int[] frequency
        int index = 0; //Allows return to index in organizedTokens when moving to next dictionary word
        for(int i = 0; i < indexedDictionary.size(); i++){ //Iterate through each dictionary word
            for(int j = index; j < organizedTokens.size(); j++){ //Iterate through every review word
                if(indexedDictionary.get(i).equals(organizedTokens.get(j))){
                    frequency[i]++;
                    if(j != organizedTokens.size() - 1) { //Checking to see we aren't at the last word
                        if (!indexedDictionary.get(i).equals(organizedTokens.get(j + 1))) {
                            index = j + 1;
                            j = organizedTokens.size();
                        }
                    }
                }
            }
        }
        //Reduce number of features (tokens to consider) by having a frequency threshold
        for(int i = 0; i < frequency.length; i++) {
            if (frequency[i] >= frequencyThreshold) {
                //Finding out the size of the double[] we'll want for a shorter frequency array
                thresholdCrossingTokens++;
            }
        }

        double[] newFrequency = new double[thresholdCrossingTokens];
        int index2 = 0;
        for(int i = 0; i < frequency.length; i++){
            if(frequency[i] >= frequencyThreshold) {
                newFrequency[index2] = frequency[i];
                newDictionary.add(indexedDictionary.get(i));
                index2++;
            }
        }

        double[] numberOfDocumentsContainingToken = new double[newDictionary.size()];
        for(int i = 0; i < newDictionary.size(); i++){ //Iterate through each dictionary word
            for (int j = 0; j < tokens.size(); j++) { //Iterate through every review
                for (int k = 0; k < tokens.get(j).length; k++) { //Iterate through every word of a review
                    if (newDictionary.get(i).equals(tokens.get(j)[k])) {
                        numberOfDocumentsContainingToken[i]++;
                        if (j != tokens.size() - 1) {
                            k = tokens.get(j).length; //Move to next review if word was already found in this review
                        }
                    }
                }
            }
        }


        for(int i = 0; i < newFrequency.length; i++){
            if(numberOfDocumentsContainingToken[i] >= 500){
                finalDictionary.add(newDictionary.get(i));
            }
        }
        /*
        for(int i = 0; i < finalDictionary.size(); i++){
            System.out.println(finalDictionary.get(i));
        }
        System.out.println("finalDictionary.size() : " + finalDictionary.size());
        */
    }

    public double[] vectorize(ArrayList<String> review){
        //tokenize review
        String[] tokenVector = new String[finalDictionary.size()];
        for(int i = 0; i < review.size(); i++){
            tokenVector = review.get(i).trim().split("\\s+");
        }
        //compare to dictionary and form numerical vector
        double[] vector = new double[finalDictionary.size()];
        for(int i = 0; i < finalDictionary.size(); i++){
            vector[i] = 0; //initialize to 0
            for(int j = 0; j < tokenVector.length; j++){
                if(tokenVector[j].equals(finalDictionary.get(i))){
                    vector[i]++;
                }
            }
        }
        return vector;
    }

    public void TFIDF(double[] newFrequency){
        //TF
        double[] tf = new double[newDictionary.size()];
        for(int i = 0; i < tf.length; i++){
            tf[i] = 0;
        }
        for(int i = 0; i < newFrequency.length; i++) { //Iterate through frequencies recorded
            tf[i] = (newFrequency[i] / (organizedTokens.size()));
        }

        //IDF
        double[] numberOfDocumentsContainingToken = new double[newDictionary.size()];
        for(int i = 0; i < newDictionary.size(); i++){ //Iterate through each dictionary word
            for (int j = 0; j < tokens.size(); j++) { //Iterate through every review
                for (int k = 0; k < tokens.get(j).length; k++) { //Iterate through every word of a review
                    if (newDictionary.get(i).equals(tokens.get(j)[k])) {
                        numberOfDocumentsContainingToken[i]++;
                        if (j != tokens.size() - 1) {
                            k = tokens.get(j).length; //Move to next review if word was already found in this review
                        }
                    }
                }
            }
        }
        double[] idf = new double[newDictionary.size()];
        for(int i = 0; i < idf.length; i++){
            idf[i] = Math.log((double)(trainingReviews.size() / (1 + numberOfDocumentsContainingToken[i])));
        }

        //TF-IDF
        double[] tfidf = new double[newDictionary.size()];
        for(int i = 0; i < tfidf.length; i++){
            tfidf[i] = tf[i] * idf[i];
        }

        inputBiases = tfidf;
    }

    public void readyTestData(String testDataLocation) throws IOException {
        File csv = new File(testDataLocation);
        FileReader fileReader = new FileReader(csv);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        //-------------------------------------------------------------------------------------------------------------
        gatherStopwords();
        for(String line; (line = bufferedReader.readLine()) != null; ) {

            ArrayList<String> testReview = new ArrayList<>();

            char[] last3 = new char[3];
            //Labels we are looking for
            char[] pos = {'p','o','s'};
            char[] neg = {'n','e','g'};
            //Go to end of line, grab last three characters, throw into char[] last3
            char[] lineCharArray = line.toCharArray();
            for(int i = 2; i >=0 ; i--)
                last3[2 - i] = lineCharArray[lineCharArray.length - (i + 1)];
            /*
            If last3 does not equal either pos or neg label, add line to current review
            (as element to our review, ArrayList<String> trainingReview)
            */
            boolean labelFound = false;
            if(Arrays.equals(last3,pos) || Arrays.equals(last3,neg))
                labelFound = true;
            if(labelFound == false)
                testReview.add(cleanText(line, stopwords));
            /*
            If last3 does equal either pos or neg label, then need both training label and
            line with training label cut off.
            */
            else{
                //Cast last3 ("pos" or "neg") into a String (the pos or neg label)
                String label = new String(last3);
                //Add label to ArrayList<String> trainingLabels
                testLabels.add(label);
                //Snip training label off end of line
                String lineWithoutLabel = line.substring(0, line.length() - 3);
                //Add line to review
                testReview.add(cleanText(lineWithoutLabel, stopwords));
            }

            //Add review to clean dataset of training reviews
            testReviews.add(testReview);

        }

        //Create correct answer vector for neural network to understand. 1 = positive, 0 = negative.
        for(int i = 0; i < testLabels.size(); i++){
            double[] answer = new double[2];
            if(testLabels.get(i).equals("pos")){
                answer[0] = 1;
                answer[1] = 0;
                testAnswers.add(answer);
            }
            else{
                answer[0] = 0;
                answer[1] = 1;
                testAnswers.add(answer);
            }

        }

        for(int i = 0; i < testReviews.size(); i++){
            double[] vector;
            vector = vectorize(testReviews.get(i));
            testMatrix.add(vector);
        }
    }
}