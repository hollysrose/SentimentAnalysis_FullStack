package edu.ucmo.spring_bare_bones;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;


public class NeuralNetwork {

    /*
    Do not worry about the attributes below initially. I have to use terms to define them
    that do not make sense outside of the code that follows.
    Much more helpful to go directly to the next block of code.
    */
    //-----------------------------------------------------------//

    //Initializing the Vectorizer
    static Vectorizer vectorizer = new Vectorizer();

    //To keep count of the number of tweets categorized correctly
    static int count = 0;

    //Amount a synapse's weight is multiplied by if changed in back()
    static final double learningRate = 0.5;
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
    static Neuron[] hiddenLayer1;

    static Neuron[] hiddenLayer2;
    /*
    Initializing the last layer of neurons, called the output layer.
    Our output layer consists of a single neuron.
    */
    static Neuron[] outputLayer;
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
        inputLayer = new Neuron[inputs];
        hiddenLayer1 = new Neuron[hidden];
        hiddenLayer2 = new Neuron[(int)Math.ceil(0.33 * hidden)];
        outputLayer = new Neuron[2];

        //Filling the input layers (an array of neurons) with neurons
        for(int i = 0; i < inputLayer.length; i++) {
            inputLayer[i] = new Neuron(0.1);
        }

        //Filling the hidden layer (an array of neurons) with neurons
        for(int i = 0; i < hiddenLayer1.length; i++) {
            hiddenLayer1[i] = new Neuron(0.1);
        }

        for(int i = 0; i < hiddenLayer2.length; i++) {
            hiddenLayer2[i] = new Neuron(0.1);
        }

        //The final layer of the neuron network. A single neuron.
        for(int i = 0; i < outputLayer.length; i++) {
            outputLayer[i] = new Neuron(0.1);
        }

        /*
        Initializing the synapses between the neurons of input layer and hidden layer.
        Each input neuron has a synapse to every neuron of the hidden layer.
        */
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer1.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapse = new Synapse(inputLayer[i], hiddenLayer1[j]);
                inputLayer[i].addRightSynapse(synapse);
                hiddenLayer1[j].addLeftSynapse(synapse);

            }
        }

        for (int i = 0; i < hiddenLayer1.length; i++) {
            for (int j = 0; j < hiddenLayer2.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapse = new Synapse(hiddenLayer1[i], hiddenLayer2[j]);
                hiddenLayer1[i].addRightSynapse(synapse);
                hiddenLayer2[j].addLeftSynapse(synapse);
            }
        }

        for (int i = 0; i < hiddenLayer2.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapse = new Synapse(hiddenLayer2[i], outputLayer[j]);
                hiddenLayer2[i].addRightSynapse(synapse);
                outputLayer[j].addLeftSynapse(synapse);
            }
        }
    }

    NeuralNetwork(int inputs, int hidden, double[] inputBiases) {

        //Setting size of each layer of neurons
        inputLayer = new Neuron[inputs];
        hiddenLayer1 = new Neuron[hidden];
        hiddenLayer2 = new Neuron[(int)Math.ceil(0.33 * hidden)];

        //Filling the input layers (an array of neurons) with neurons
        for(int i = 0; i < inputLayer.length; i++) {
            inputLayer[i] = new Neuron(inputBiases[i]);
        }

        //Filling the hidden layer (an array of neurons) with neurons
        for(int i = 0; i < hiddenLayer1.length; i++) {
            hiddenLayer1[i] = new Neuron();
        }

        for(int i = 0; i < hiddenLayer2.length; i++) {
            hiddenLayer2[i] = new Neuron();
        }

        //The final layer of the neuron network. A single neuron.
        for(int i = 0; i < 2; i++) {
            outputLayer[i] = new Neuron();
        }

        /*
        Initializing the synapses between the neurons of input layer and hidden layer.
        Each input neuron has a synapse to every neuron of the hidden layer.
        */
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer1.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapses = new Synapse(inputLayer[i], hiddenLayer1[j], inputBiases[i]);
                inputLayer[i].addRightSynapse(synapses);
                hiddenLayer1[j].addLeftSynapse(synapses);

            }
        }

        for (int i = 0; i < hiddenLayer1.length; i++) {
            for (int j = 0; j < hiddenLayer2.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapses = new Synapse(hiddenLayer1[i], hiddenLayer2[j]);
                hiddenLayer1[i].addRightSynapse(synapses);
                hiddenLayer2[j].addLeftSynapse(synapses);
            }
        }

        for (int i = 0; i < hiddenLayer2.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapses = new Synapse(hiddenLayer2[i], outputLayer[j]);
                hiddenLayer2[i].addRightSynapse(synapses);
                outputLayer[j].addLeftSynapse(synapses);
            }
        }
    }

    /*
    The forward() method is forward propagation, and it is used in training and in testing.
    */
    public double[] forward(double[] vector) {

        ////////////////////////////////////
        System.out.println("Input vector :");
        for(int n = 0; n < vector.length; n++) {
            System.out.print(vector[n] + " ");
        }
        System.out.println("\n");

        /*
        Each input layer (first layer) neuron has an output assigned as the value of its respective
        WordVectorizer vector element.
        In other words, each input layer neuron is assigned the respective positive or negative value
        of an individual word within the input.
        */
        for (int i = 0; i < inputLayer.length; i++) {
            inputLayer[i].calculateOutput(vector[i]);
        }
        ////////////////////////////////////
        System.out.println("inputLayer :");
        for(int n = 0; n < inputLayer.length; n++) {
            System.out.print(inputLayer[n].output + " ");
        }
        System.out.println("\n");

        /*
        Each hidden layer neuron receives the outputs of neurons of the previous layer.
        (Our neural network has only one hidden layer, so the "previous layer" of the hidden layer
        is always the input layer.)
        Each hidden layer neuron takes these outputs, multiplies each output by the weight
        of the synapse by which it was received, and adds the products. This is done in
        the hiddenNeuronOutput() of Neuron.
        */
        for (int i = 0; i < hiddenLayer1.length; i++) {
            hiddenLayer1[i].calculateOutput();
        }
        ////////////////////////////////////
        System.out.println("hiddenLayer1 :");
        for(int n = 0; n < hiddenLayer1.length; n++) {
            System.out.print(hiddenLayer1[n].output + " ");
        }
        System.out.println("\n");

        for (int i = 0; i < hiddenLayer2.length; i++) {
            hiddenLayer2[i].calculateOutput();
        }
        ////////////////////////////////////
        System.out.println("hiddenLayer2 :");
        for(int n = 0; n < hiddenLayer2.length; n++) {
            System.out.print(hiddenLayer2[n].output + " ");
        }
        System.out.println("\n");

        /*
        Calculating the output of our singular output layer neuron.
        */
        double[] analysis = new double[2];
        for(int i = 0; i < outputLayer.length; i++) {
            outputLayer[i].calculateOutput();
            analysis[i] = outputLayer[i].output;
        }

        //Reset all neurons outputs to 0;
        //Do not want to do this the last forward before a back(), since need to refer to output

        return analysis;
    }


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
    public void back(double[] averageLoss) {

        ArrayList<Double> dAverageLoss_dNetInput = new ArrayList<>();

        double[] weightsUpdate_o = gradient_o(averageLoss, dAverageLoss_dNetInput);

        double[] weightsUpdate_h2 = gradient_h(averageLoss, hiddenLayer1, hiddenLayer2, outputLayer, dAverageLoss_dNetInput);

        double[] weightsUpdate_h1 = gradient_h(averageLoss, inputLayer, hiddenLayer1, hiddenLayer2, dAverageLoss_dNetInput);

        System.out.print("weightsUpdate_o:");
        for(int i = 0; i < weightsUpdate_o.length; i++) {
            System.out.print(weightsUpdate_o[i] + " ");
        }
        System.out.print("\n");
        System.out.print("weightsUpdate_h2:");
        for(int i = 0; i < weightsUpdate_h2.length; i++) {
            System.out.print(weightsUpdate_h2[i] + " ");
        }
        System.out.print("\n");
        System.out.print("weightsUpdate_h1:");
        for(int i = 0; i < weightsUpdate_h1.length; i++) {
            System.out.print(weightsUpdate_h1[i] + " ");
        }
        System.out.print("\n");

        //update weights
        //Between output and hidden2
        for(int outputNeuron = 0; outputNeuron < outputLayer.length; outputNeuron++){
            for(int hidden2Neuron = 0; hidden2Neuron < hiddenLayer2.length; hidden2Neuron++){
                outputLayer[outputNeuron].getLeftSynapses().get(hidden2Neuron).updateWeight(weightsUpdate_o[hidden2Neuron], learningRate);
            }
        }
        for(int hidden2Neuron = 0; hidden2Neuron < hiddenLayer2.length; hidden2Neuron++){
            for(int outputNeuron = 0; outputNeuron < outputLayer.length; outputNeuron++){
                hiddenLayer2[hidden2Neuron].getRightSynapses().get(outputNeuron).updateWeight(weightsUpdate_o[outputNeuron], learningRate);
            }
        }

        System.out.println("Between input and h1:");
        for(int i = 0; i < hiddenLayer1.length; i++){
            System.out.print(inputLayer[i].getRightSynapses().get(i).weight + " ");
        }

        //Between hidden2 and hidden1
        for(int hidden2Neuron = 0; hidden2Neuron < hiddenLayer2.length; hidden2Neuron++){
            for(int hidden1Neuron = 0; hidden1Neuron < hiddenLayer1.length; hidden1Neuron++){
                hiddenLayer2[hidden2Neuron].getLeftSynapses().get(hidden1Neuron).updateWeight(weightsUpdate_h2[hidden1Neuron], learningRate);
            }
        }
        for(int hidden1Neuron = 0; hidden1Neuron < hiddenLayer1.length; hidden1Neuron++){
            for(int hidden2Neuron = 0; hidden2Neuron < hiddenLayer2.length; hidden2Neuron++){
                hiddenLayer1[hidden1Neuron].getRightSynapses().get(hidden2Neuron).updateWeight(weightsUpdate_h2[hidden2Neuron], learningRate);
            }
        }

        System.out.println("Between h1 and h2:");
        for(int i = 0; i < hiddenLayer2.length; i++){
            System.out.print(hiddenLayer1[i].getRightSynapses().get(i).weight + " ");
        }

        //Between hidden1 and input
        for(int hidden1Neuron = 0; hidden1Neuron < hiddenLayer1.length; hidden1Neuron++){
            for(int inputNeuron = 0; inputNeuron < inputLayer.length; inputNeuron++){
                hiddenLayer1[hidden1Neuron].getLeftSynapses().get(inputNeuron).updateWeight(weightsUpdate_h1[inputNeuron], learningRate);
            }
        }
        for(int inputNeuron = 0; inputNeuron < inputLayer.length; inputNeuron++){
            for(int hidden1Neuron = 0; hidden1Neuron < hiddenLayer1.length; hidden1Neuron++){
                inputLayer[inputNeuron].getRightSynapses().get(hidden1Neuron).updateWeight(weightsUpdate_h1[hidden1Neuron], learningRate);
            }
        }

        System.out.println("Between h2 and output:");
        for(int i = 0; i < outputLayer.length; i++){
            System.out.print(hiddenLayer2[i].getRightSynapses().get(i).weight + " ");
        }


    }

    public double[] gradient_o(double[] averageLoss, ArrayList<Double> dAverageLoss_dNetInput){

        double[] weightsUpdate =  new double [outputLayer.length * hiddenLayer2.length];

        //dAverageLoss_dSynapse = dAverageLoss_dOutput * dOutput_dNetInput * dNetInput_dSynapse
        double dAverageLoss_dSynapse = 0;
        double dAverageLoss_dOutput = 0;
        double dOutput_dNetInput = 0;
        double dNetInput_dSynapse = 0;

        int weightsIndex = 0;
        int idealIndex = 2;
        for(int neuron = 0; neuron < outputLayer.length; neuron++){
            //dAverageLoss_dOutput = -(ideal - outputLayer[neuron].output) = (outputLayer[neuron].output - ideal)
            //However, we are taking the average, so we pull from averageLoss


            dAverageLoss_dOutput = averageLoss[neuron] - averageLoss[idealIndex];

            //dOutput_dNetInput = derivativeReLU((N-1)o)
            dOutput_dNetInput = outputLayer[neuron].derivativeReLU(outputLayer[neuron].output);

            //Provide references for gradient calculations of other layers
            dAverageLoss_dNetInput.add(dAverageLoss_dOutput * dOutput_dNetInput);

            for(int leftNeuron = 0; leftNeuron < hiddenLayer2.length; leftNeuron++){
                //dNetInput_dSynapse = hiddenLayer2[leftNeuron].output
                dNetInput_dSynapse = hiddenLayer2[leftNeuron].output;

                dAverageLoss_dSynapse = dAverageLoss_dOutput * dOutput_dNetInput * dNetInput_dSynapse;

                //update synapse weight array
                weightsUpdate[weightsIndex + leftNeuron] = dAverageLoss_dSynapse;

            }
            idealIndex = 3;
            weightsIndex = hiddenLayer2.length;
        }

        return weightsUpdate;
    }

    public double[] gradient_h(double[] averageLoss,
                               Neuron[] leftLayer, Neuron[] thisLayer, Neuron[] rightLayer,
                               ArrayList<Double> dAverageLoss_dNetInput) {

        double[] weightsUpdate =  new double [thisLayer.length * leftLayer.length];

        //dAverageLoss_dSynapse = dAverageLoss_dOutput * dOutput_dNetInput * dNetInput_dSynapse
        double dAverageLoss_dSynapse = 0;
        double dAverageLoss_dOutput = 0;
        double dOutput_dNetInput = 0;
        double dNetInput_dSynapse = 0;

        int weightsIndex = 0;
        //int weightsIndexMultiplier = 1;

        for(int neuron = 0; neuron < thisLayer.length; neuron++){
            //dAverageLoss_dOutput = -(ideal - outputLayer[neuron].output) = (outputLayer[neuron].output - ideal)
            //However, we are taking the average, so we pull from averageLoss

            //dOutput_dNetInput = derivativeReLU((N-1)o)
            dOutput_dNetInput = thisLayer[neuron].derivativeReLU(thisLayer[neuron].output);

            //for(int rightNeuron = 0; rightNeuron < rightLayer.length; rightNeuron++) {
            int rightNeuron = 0;
            while(rightNeuron < rightLayer.length){

                //Take weightsUpdate_o iteration, divide by dNetInput_dSynapse
                dAverageLoss_dOutput += dAverageLoss_dNetInput.get(rightNeuron) * thisLayer[neuron].getRightSynapses().get(rightNeuron).weight;
                rightNeuron++;
            }
            //Provide references for gradient calculations of other layers
            dAverageLoss_dNetInput.add(dAverageLoss_dOutput * dOutput_dNetInput);

            for (int leftNeuron = 0; leftNeuron < leftLayer.length; leftNeuron++) {
                //dNetInput_dSynapse = hiddenLayer2[leftNeuron].output
                dNetInput_dSynapse = leftLayer[leftNeuron].output;

                dAverageLoss_dSynapse = dAverageLoss_dOutput * dOutput_dNetInput * dNetInput_dSynapse;

                //update synapse weight array
                weightsUpdate[weightsIndex/* + leftNeuron*/] = dAverageLoss_dSynapse;
                ///
                weightsIndex++;
            }
            //weightsIndex = leftLayer.length * weightsIndexMultiplier;
            //weightsIndexMultiplier++;
            dAverageLoss_dOutput = 0;

        }

        //Removing references of last layer
        for(int i = 0; i < rightLayer.length; i++){
            dAverageLoss_dNetInput.remove(i);
        }

        return weightsUpdate;
    }

    public void train(int epochs){
        double[] averageLoss = new double[5];
        for(int i = 0; i < averageLoss.length; i++){
            averageLoss[i] = 0.0;
        }

        //For determining stochastic gradient descent period (back-propagation initiated every period)
        int backPropNumber = 1;
        //To time period from a forward-feed to end of a back-propagation
        double backPropTime = 0.0;
        //Per epoch...
        int epochNumber = 1;
        for(int i = 0; i < epochs; i++){
            //Timing
            long epochStartTime = System.currentTimeMillis();
            long backPropStartTime = System.currentTimeMillis();
            //Iterate through training vectors
            for(int j = 0; j < /*vectorizer.trainingMatrix.size()*/ 5; j++) {
                double[] analysis = forward(vectorizer.trainingMatrix.get(j));
                System.out.println("\n");
                double[] outputNeuronLoss = new double[2];
                outputNeuronLoss[0] = Math.pow((vectorizer.trainingAnswers.get(j)[0] - analysis[0]),2);
                outputNeuronLoss[1] = Math.pow((vectorizer.trainingAnswers.get(j)[1] - analysis[1]),2);
                double loss = outputNeuronLoss[0] + outputNeuronLoss[1];
                System.out.println("Top: " + analysis[0] + ", Bottom: " + analysis[1]);
                System.out.println("Answers: " + vectorizer.trainingAnswers.get(j)[0] + " " + vectorizer.trainingAnswers.get(j)[1]);

                averageLoss[4] += loss;
                averageLoss[3] += vectorizer.trainingAnswers.get(j)[1];
                averageLoss[2] += vectorizer.trainingAnswers.get(j)[0];
                averageLoss[1] += analysis[1];
                averageLoss[0] += analysis[0];

                backPropTime++;
                //Stochastic gradient descent every 5000 training reviews throughout the epochs
                if((backPropTime == 2.00) || (j == vectorizer.trainingMatrix.size() - 1)){

                    for(int k = 0; k < averageLoss.length; k++){
                        averageLoss[k] = averageLoss[k] / backPropTime;
                    }

                    backPropTime = 0.0;
                    back(averageLoss);
                    long backPropEndTime = System.currentTimeMillis();
                    long backPropRoundTime = (backPropEndTime - backPropStartTime)/1000;
                    //System.out.println("Back propagation " + backPropNumber + " took " + backPropRoundTime + " seconds");

                    backPropStartTime = System.currentTimeMillis();

                    backPropNumber++;

                    Arrays.fill(averageLoss, 0.0);
                }

                //Zero out all outputs to clean the neural network
                for(int z = 0; z < inputLayer.length; z++){
                    inputLayer[z].output = 0;
                }
                for(int z = 0; z < hiddenLayer1.length; z++){
                    hiddenLayer1[z].output = 0;
                }
                for(int z = 0; z < hiddenLayer2.length; z++){
                    hiddenLayer2[z].output = 0;
                }
                for(int z = 0; z < outputLayer.length; z++){
                    outputLayer[z].output = 0;
                }
            }
            long epochEndTime = System.currentTimeMillis();
            long epochTime = (epochEndTime - epochStartTime)/(1000 * 60);
            System.out.println("Epoch " + epochNumber + " took " + epochTime + " minutes, or " + epochTime * 60000 + " milliseconds");
            epochNumber++;
        }
    }

    public void test(){

        double[] avgLossNumerators = new double[25000];

        int correct = 0;

        for(int i = 0; i < vectorizer.testMatrix.size(); i++) {
            double[] analysis = forward(vectorizer.testMatrix.get(i));
            double[] outputNeuronLoss = new double[2];
            outputNeuronLoss[0] = Math.pow((vectorizer.testAnswers.get(i)[0] - analysis[0]),2);
            outputNeuronLoss[1] = Math.pow((vectorizer.testAnswers.get(i)[1] - analysis[1]),2);
            double loss = outputNeuronLoss[0] + outputNeuronLoss[1];

            avgLossNumerators[i] = loss;

            double actual = Math.max(analysis[0], analysis[1]);

            int actualIndex;
            if(actual  == analysis[0]) {
                actualIndex = 0;
            }
            else{
                actualIndex = 1;
            }
            double answer = Math.max(vectorizer.testAnswers.get(i)[0],vectorizer.testAnswers.get(i)[1]);

            int answerIndex;
            if(answer  == vectorizer.testAnswers.get(i)[0]) {
                answerIndex = 0;
            }
            else{
                answerIndex = 1;
            }

            if((actualIndex == answerIndex)){
                correct++;
            }
        }

        double sum = 0;
        for(int i = 0; i < avgLossNumerators.length; i++){
            sum += avgLossNumerators[i];
        }
        double avgLoss = sum / 25000.00;

        System.out.println("Average Loss: " + avgLoss);
        System.out.println("Percent correct: " + (correct/*/vectorizer.testMatrix.size())/250*/));

    }


    public static void main(String[] args) throws IOException, InterruptedException {

        //Training
        //-----------------------------------------------------------//
        //System.out.println("Training Begins. Teach me.\n");

        //WordVectorizer produces an input, a vector called bagOfWords, for our neural network.
        vectorizer.readyTrainingData("src/main/resources/train.csv");


        /*
        Initializing our neural network.
        The number of input layer neurons will be wv.bagOfWords.size().
        (This is one input layer neuron for each word of text deemed valuable.)
        There will be one hidden layer of 700 neurons.
        */
        NeuralNetwork Ultron = new NeuralNetwork(vectorizer.finalDictionary.size(), 20);

        long startTime = System.currentTimeMillis();

        System.out.println("I wanna learn!\n");

        //We train the network for 5 epochs.
        Ultron.train(1);


        System.out.println("I am ready!");
        //-----------------------------------------------------------//


        //Testing
        //-----------------------------------------------------------//
        System.out.println("I have surpassed you, humans. Gather the test.");


        /*
        //WordVectorizer takes out test data as the input for our neural network.
        vectorizer.readyTestData("src/resources/test.csv");

        Ultron.test();

        long endTime = System.currentTimeMillis();

        long time =  (endTime - startTime)/(1000 * 60);

        System.out.println("Process took " + time + " minutes.");
        */


        /*
        We have 600 test items.
        Our neural network will go through each item, label each item, and
        report the difference in the score (sentiment) it assigned and the value
        given (correct sentiment) as the test datum's label.
        */

        /*
        for(int i = 0; i < 25000; i++) {
            double error = 0.5*Math.pow(Ultron.forward(vectorizer.testMatrix.get(i))-vectorizer.testAnswers, 2);

            System.out.printf("---> %.2f\n", error);



            Keeping count of how many times the neural network produced an answer
            in an acceptable range. Let's choose an error of 20% or less as
            having been accurate.

            if(error <= 0.2) {
                count++;
            }


        }


        //Measuring and reporting the accuracy that the neural network classified sentiment.
        System.out.println("ACCURACY: " + (count/250)+ "%");
    */
        //-----------------------------------------------------------//
    }

}