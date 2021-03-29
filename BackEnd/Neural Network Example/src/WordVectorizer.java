import java.io.*;
import java.util.*;

/*
The textMatrix interface below defines method signatures that all text-to-vector classes (every kind of neural network
architecture we might use) needs to implement.
The interface will allow us to transition between architectures of neural networks smoothly.
*/
interface textMatrix {

    String TextCleaner(String line);

    void bagOfWordsCreator(String line);

    ArrayList<Integer> wordVectorizer(String line);

}

/*
The WordVectorizer is the text-to-vector class we use for our "bare and basic" neural network.
We might make a different WordVectorizer for another neural network architecture.
*/
public class WordVectorizer implements textMatrix {

    public ArrayList<Integer> label_array = new ArrayList<Integer>();
    public ArrayList<Integer> test_label_array = new ArrayList<Integer>();
    public ArrayList<ArrayList<Integer>> test_matrix = new ArrayList<ArrayList<Integer>>();
    public ArrayList<ArrayList<Integer>> input_matrix = new ArrayList<ArrayList<Integer>>();

    //This will be used to initially store all words
    ArrayList<String> wordsList = new ArrayList<String>();

    //Used to store "stopwords" (will be defined in stopWordsReader() method soon, below
    ArrayList<String> stopWords = new ArrayList<String>();

    //Used to store words by their frequency
    public Set<String> bagOfWords = new HashSet<String>();

    //lines and test_lines store positive and negative tweets, respectively
    static ArrayList<String> lines = new ArrayList<String>();
    static ArrayList<String> test_lines = new ArrayList<String>();

    //-----------------------------------------------------------//
    /*
    The following methods are used by the wordVectorizer() interface method. These following methods are implementations
    specifically for a basic neural network (as opposed to a convoluted neural network (CNN) or other architecture).
    */
    //-----------------------------------------------------------//

    /*
    The stopWordsReader() method finds words that do not add sentimental meaning (i.e. "a", "the", etc.)
    by comparing words in the text to our .txt of words we consider to be "stopwords". The found stopwords
    are stored.
    */
    static void stopWordsReader(ArrayList<String> ar, String path) throws IOException {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader bfr = new BufferedReader(fr);

        String stopWords = "";

        while((stopWords=bfr.readLine()) != null) {
            ar.add(stopWords);
        }
        bfr.close();
    }

    /*
    The TextCleaner() method below removes "RT" (retweet symbol), https links, and more unnecessary characters
    that are not words in our stopwords.txt. We could add them to stopwords.txt and remove this method.
    */
    public String TextCleaner(String text) {
        return text.replaceAll("[!#$%^&*?,.;?\"0-9/;():-]", "").replace("RT", "")
                .replaceAll("http.*?\\s", "").replaceAll("@.*?\\s", "").replaceAll("www.*?\\s", "")
                .replace("quot", "").replace("amp", "");
    }

    /*
    The bagOfWordsCreator() method splits each line of text into "tokens" (pieces of text to which we will
    attribute sentiment and adds these tokens to the (Set type field) bagOfWords to obtain a vocabulary of unique
    words. The method also adds them to (ArrayList type field) wordsList to obtain the frequency of each word
    (to ascertain how much of an impact each word ("token") should have in measuring sentiment).
    */
    public void bagOfWordsCreator(String line){
        String[] tokens = line.trim().split("\\s+");

        for(int i=0; i < tokens.length; i++) {
            bagOfWords.add(tokens[i]);
            wordsList.add(tokens[i]);
        }
    }

    /*
    The wordRemover() method removes the stopwords we've found, as well as highly infrequent words.
    (in the hope to get rid of "hand", "sandwich", but keep the more frequent "like", "love, "want", "hate").
    */
    public void wordRemover() {
        for(int i=0; i < wordsList.size(); i++) {
            if((Collections.frequency(wordsList, wordsList.get(i)) <= 2) || wordsList.get(i).equals(" ")==true) {
                bagOfWords.remove(wordsList.get(i));
            }
        }

        bagOfWords.removeAll(stopWords);
    }

    /*
    The wordVectorizer() method converts each line of text into a vector of numbers. We feed this vector
    into our neural network. This vectorizer forms a binary vector--meaning, when our neural network
    reads some new text (not the test sample), each word ("token") is assigned either 1
    (indicating presence of a word in our bagOfWords) or 0 (indicating absence).
    This is a poor way to measure.
    */
    public ArrayList<Integer> wordVectorizer(String line) {
        String[] tweet = line.split("\\s+");

        //Initializes a vector with length of bagOfWords and populates it with zeroes.
        ArrayList<Integer> vector = new ArrayList<Integer>(Collections.nCopies(bagOfWords.size(),0));

        for(int i=0; i < tweet.length; i++) {
            if(bagOfWords.contains(tweet[i])) {
                vector.set(new ArrayList<String>(bagOfWords).indexOf(tweet[i]), 1 );
            }
        }

        return vector;
    }

    /*
    The trainDataReader() method reads text line by line, cleans and tokenizes each line, creates a bagOfWords.
    */
    public void trainDataReader(String train_path) throws IOException {
        stopWordsReader(stopWords, "src/resources/stopwords.txt");

        //Creating File Object to hold our .csv training data.
        File csv = new File(train_path);

        FileReader fr = new FileReader(csv);

        BufferedReader bfr = new BufferedReader(fr);

        try {
            System.out.println("TWEETS\n");
            for(String line; (line=bfr.readLine())!= null;) {

                lines.add(TextCleaner(line.split(",")[5].toLowerCase()));

                //input_matrix.add(wordVectorizer(TextCleaner(line.split(",")[5].toLowerCase())));

                label_array.add(Integer.parseInt(line.split(",")[0]));

                bagOfWordsCreator(TextCleaner(line.split(",")[5].toLowerCase()));

                System.out.println(TextCleaner(line.split(",")[5].toLowerCase()));
            }

            wordRemover();

            for(String l: lines) {
                input_matrix.add(wordVectorizer(l));
            }
        }
        catch(IOException ioe){
            System.err.println("Unexpected issue while reading:\n\n" + ioe);
        }
        catch(Exception e){
            System.err.println("Unexpected general issue:\n\n" + e);
        }

        //Closing the FileReader and BufferedReader Objects
        finally {
            try{
                fr.close();
                bfr.close();
            }

            catch(Exception e) {
                System.err.println("Exception while closing: " + e);
            }
        }
    }

    /*
    The testDataReader() method is similar to trainDataReader(), except that testDataReader() does not have
    the aim to create anything--no bagOfWords is created. We use what we learned from trainDataReader()
    as our standard, our comparison, our ruler, for any data read by testDataReader.
    */
    public void testDataReader(String test_path)throws IOException {
        File csv = new File(test_path);

        FileReader fr = new FileReader(csv);

        BufferedReader bfr = new BufferedReader(fr);

        System.out.println("TWEETS\n");
        for(String line; (line=bfr.readLine())!= null;) {
            test_lines.add(TextCleaner(line.split(",")[5].toLowerCase()));

            test_label_array.add(Integer.parseInt(line.split(",")[0]));

            System.out.println(TextCleaner(line.split(",")[5].toLowerCase()));
        }

        for(String l: test_lines) {
            test_matrix.add(wordVectorizer(l));
        }
    }
}