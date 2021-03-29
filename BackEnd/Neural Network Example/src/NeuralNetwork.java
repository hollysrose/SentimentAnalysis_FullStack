import java.io.IOException;
import java.util.*;


public class NeuralNetwork {

    /*
    Do not worry about the attributes below initially. I have to use terms to define them
    that do not make sense outside of the code that follows.
    Much more helpful to go directly to the next block of code.
    */
    //-----------------------------------------------------------//
    //We may "place a thumb on the scale" at each layer of neurons.
    double inputLayerBias = 0.0;
    double hiddenLayerBias = 0.0;

    //Initializing the WordVectorizer
    static WordVectorizer wv = new WordVectorizer();

    //To keep count of the number of tweets categorized correctly
    static int count =1;

    //Amount a synapse's weight is multiplied by if changed in back()
    static final double learningRate = 0.08;
    //-----------------------------------------------------------//

    //Initializing our neurons in their respective layers
    //-----------------------------------------------------------//
    /*
    Initializing the first layer of neurons, called the input layer.
    */
    static Neuron[] inputLayer;
    /*
    Initializing the middle layers of neurons (not first and not last), called a hidden layer.
    We have only one hidden layer.
    */
    static Neuron[] hiddenLayer;
    /*
    Initializing the last layer of neurons, called the output layer.
    Our output layer consists of a single neuron.
    */
    static Neuron outputNeuron;
    //-----------------------------------------------------------//

    /*
    Constructor.
    The integer parameter 'input' is wv.bagOfWords.size(), which is the size
    of the vector of the output of WordVectorizer. Therefore, there is an
    input neuron for every element of the input vector. In other words, there is
    an input neuron for each word the WordVectorizer output.
    The integer parameter 'hidden' is simply a determination of how many
    neurons we'd like the hidden layer to have.
    */
    NeuralNetwork(int inputs, int hidden) {

        //Setting size of each layer of neurons
        inputLayer = new Neuron[inputs + 1];
        hiddenLayer = new Neuron[hidden + 1];

        //Filling the input layers (an array of neurons) with neurons
        for(int i=0; i < inputs; i++) {
            inputLayer[i] = new Neuron();
        }
        //The last neuron of the input layer assigned as a bias neuron
        inputLayer[inputs] = new Neuron(inputLayerBias);

        //Filling the hidden layer (an array of neurons) with neurons
        for(int i=0; i < hidden; i++) {
            hiddenLayer[i] = new Neuron();
        }
        //The last neuron of the hidden layer assigned as a bias neuron
        hiddenLayer[hidden] = new Neuron(hiddenLayerBias);

        //The final layer of the neuron network. A single neuron.
        outputNeuron = new Neuron();

        /*
        Initializing the synapses between the neurons of input layer and hidden layer.
        Each input neuron has a synapse to every neuron of the hidden layer.
        */
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer.length; j++) {
                //A synapse is created and attached by each end.
                Synapses s = new Synapses(inputLayer[i],hiddenLayer[j]);
                inputLayer[i].addSynapses(s);
                hiddenLayer[j].addSynapses(s);

            }
        }


        /*
        Initializing the synapses between the neurons of hidden layer and output layer.
        Each hidden layer neuron has a synapse to every neuron of the output layer.
        Since our output layer has only one neuron, we only need one for-loop to go through
        the hidden layer neurons.
        */
        for (int i = 0; i < hiddenLayer.length; i++) {
            //A synapse is created and attached by each end.
            Synapses s = new Synapses(hiddenLayer[i] ,outputNeuron);
            hiddenLayer[i].addSynapses(s);
            outputNeuron.addSynapses(s);
        }

    }

    /*
    The forward() method is forward propagation, and it is used in training and in testing.
    */
    public double forward(ArrayList<Integer> inputArray) {
        /*
        Each input layer (first layer) neuron has an output assigned as the value of its respective
        WordVectorizer vector element.
        In other words, each input layer neuron is assigned the respective positive or negative value
        of an individual word within the input.
        */
        for (int i = 0; i < inputArray.size(); i++) {
            inputLayer[i].output = inputArray.get(i);
        }

        /*
        Each hidden layer neuron receives the outputs of neurons of the previous layer.
        (Our neural network has only one hidden layer, so the "previous layer" of the hidden layer
        is always the input layer.)
        Each hidden layer neuron takes these outputs, multiplies each output by the weight
        of the synapse by which is was received, and adds the products. This is done in
        the hiddenNeuronOutput() of Neuron.
        */
        for (int i = 0; i < hiddenLayer.length-1; i++) {
            hiddenLayer[i].hiddenNeuronOutput();
        }

        /*
        Calculating the output of our singular output layer neuron.
        */
        outputNeuron.neuronOutput();

        return (outputNeuron.output);
    }


    /////////////////////////////////////////////////
    static int binaryOutput(double y) {

        //Since tanh always has an output between -1 and 1
        return y < 0 ? 0 : 1;
    }
    /////////////////////////////////////////////////
/*
    static int textualOutput(double y) {


        return y < 0 ? "Negative":"Positive";

    }
 */
    //////////////////////////////////////
    static String textualOutput(double y) {


        return y < 0 ? "Negative":"Positive";

    }
    //////////////////////////////////////

    /*
    The back() method is back propagation, and it is only used when training the neural network. This is how
    the neural network adjusts, learns.
    Back propagation adjusts the weights of the synapses between neurons. This is the learning.
    Back propagation begins with the synapses closest to the output layer, working back
    toward the first layer of neurons.
    Notice: back() begins by calling forward(). So, We will only call back() in main().
    In main(), we will call back() a number of iterations (called 'epochs').
    back() calls forward() and then proceeds with the back() method.
    We call back() again, which begins by calling forward()...
    Picture echolocation between a bat and a cave wall. forward() and back().
    */
    public double back(ArrayList<Integer>inputs, int correctAnswer) {
        //Assigning forwardResult the return value of foward()
        double forwardResult = forward(inputs);
        /*
        An error value to determine distance between the forwardResult value (neural network's answer)
        and the answer we assigned the training datum.
        Think of this as flashcards.
        */
        double error = (forwardResult - correctAnswer);

        /*
        For each hidden layer neuron, we adjust the weight of its synapse to the output layer neuron.
        The amount by which we adjust the weight is determined by the learningRate value we assigned
        the NeuralNetwork Object (attribute declared at top) and the size of the error.
        We also multiply by the derivative of tanh.
        */
        for(int i=0; i < hiddenLayer.length; i++) {
            //Calculating the adjusted weight.
            double adjustedWeight = learningRate * error * hiddenLayer[i].output * (1-Math.pow(forwardResult,2));

            //Other ways we tried this calculation.
            //double tweakedWeight =error*(output)*(forwardResult)*(1-forwardResult);
            //double tweakedBias = error*(forwardResult)*(1-forwardResult);

            //Assigning the synapse's weight to the adjusted weight.
            hiddenLayer[i].synapses.get(i).updateWeight(adjustedWeight);
        }


        /*
        Adjusting the weight of the synapses between the hidden layer neurons and input layer neurons.
        The reason we having a different block of code for these synapses as opposed to the synapses
        addressed in the block of code above (synapses between output layer and hidden layer) is because
        we have only one output neuron. We don't need two for-loops in that case. Here, we do.
        Another difference is that, to calculate the adjusted weight of a synapse, we multiply the
        derivative of the rectified linear unit (ReLU) rather than the derivative of tanh, as in the block
        of code above.
        */
        for (int i = 0; i < inputLayer.length; i++) {
            for(int j=0; j < hiddenLayer.length; j++) {
                //Calculating the adjusted weight.
                double adjustedWeight = learningRate * error * inputLayer[i].output * Neuron.derivativeReLU(inputLayer[i].output);

                //Might use this:
                //double deltaBias = error*Neuron_Object.relu_deriv_func(neuron.output);

                //Assigning the synapse's weight to the adjusted weight.
                inputLayer[i].synapses.get(j).updateWeight(adjustedWeight);
            }
        }

        return forwardResult;
    }


    public static void main(String[] args) throws IOException, InterruptedException {

        //Training
        //-----------------------------------------------------------//
        System.out.println("Training Begins. Teach me.\n");

        //WordVectorizer produces an input, a vector called bagOfWords, for our neural network.
        wv.trainDataReader("src/resources/training_Tweets.csv");

        /*
        Initializing our neural network.
        The number of input layer neurons will be wv.bagOfWords.size().
        (This is one input layer neuron for each word of text deemed valuable.)
        There will be one hidden layer of 30 neurons.
        */
        NeuralNetwork Ultron = new NeuralNetwork(wv.bagOfWords.size() , 30);

        //We train the network for 100 epochs.
        for(int j=0; j < 100; j++) {
            for(int i=0; i <6000; i++) {
                Ultron.back(wv.input_matrix.get(i), wv.label_array.get(i));
            }
        }

        System.out.println("I am ready!\n");
        //-----------------------------------------------------------//


        //Testing
        //-----------------------------------------------------------//
        System.out.println("I have surpassed you, humans. Test me.");

        //WordVectorizer takes out test data as the input for our neural network.
        wv.testDataReader("src/resources/testTweets(600).csv");

        /*
        We have 600 test items.
        Our neural network will go through each item, label each item, and
        report the difference in the score (sentiment) it assigned and the value
        given (correct sentiment) as the test datum's label.
        */
        for(int i=0; i < 600; i++) {
            double error = 0.5*Math.pow(Ultron.forward(wv.test_matrix.get(i))-wv.test_label_array.get(i), 2);

            System.out.printf("---> %.2f\n", error);

            /*
            Keeping count of how many times the neural network produced an answer
            in an acceptable range. Let's choose an error of 20% or less as
            having been accurate.
            */
            if(error <= 0.2) {
                count++;
            }
        }

        //Measuring and reporting the accuracy that the neural network classified sentiment.
        System.out.println("ACCURACY: " + (count/6)+ "%");
        //-----------------------------------------------------------//
    }

}