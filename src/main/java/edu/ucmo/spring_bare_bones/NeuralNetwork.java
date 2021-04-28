package edu.ucmo.spring_bare_bones;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

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
    //static final double learningRate = 0.1;
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
    static Neuron[] outputLayer;

    ArrayList<Double> Losses = new ArrayList<Double>();
    ArrayList<Long> gradientDescentTimes = new ArrayList<>();

    static ArrayList<double[]> knawledgeMatrix = new ArrayList<>();
    static ArrayList<double[]> knawledgeAnswersMatrix = new ArrayList<>();
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
        hiddenLayer = new Neuron[hidden];
        outputLayer = new Neuron[1];

        //Filling the input layers (an array of neurons) with neurons
        for(int i = 0; i < inputLayer.length; i++) {
            inputLayer[i] = new Neuron(0.0);
        }

        //Filling the hidden layer (an array of neurons) with neurons
        for(int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayer[i] = new Neuron(0.0);
        }


        //The final layer of the neuron network. A single neuron.
        for(int i = 0; i < outputLayer.length; i++) {
            outputLayer[i] = new Neuron(0.0);
        }

        /*
        Initializing the synapses between the neurons of input layer and hidden layer.
        Each input neuron has a synapse to every neuron of the hidden layer.
        */
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapse = new Synapse(inputLayer[i], hiddenLayer[j]);
                inputLayer[i].addSynapse(synapse);
                hiddenLayer[j].addSynapse(synapse);
            }
        }


        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                //A synapse is created and attached by each end.
                Synapse synapse = new Synapse(hiddenLayer[i], outputLayer[j]);
                hiddenLayer[i].addSynapse(synapse);
                outputLayer[j].addSynapse(synapse);
            }
        }
    }

    public double[] forward(double[] vector) {

        ////////////////////////////////////
        /*
        System.out.println("Input vector :");
        for(int n = 0; n < vector.length; n++) {
            System.out.print(vector[n] + " ");
        }
        System.out.println("\n");
        */

        /*
        Each input layer (first layer) neuron has an output assigned as the value of its respective
        WordVectorizer vector element.
        In other words, each input layer neuron is assigned the respective positive or negative value
        of an individual word within the input.
        */

        for (int i = 0; i < inputLayer.length; i++) {
            inputLayer[i].calculateOutput(vector[i]);
        }
        /*
        System.out.println("inputLayer :");
        for(int n = 0; n < inputLayer.length; n++) {
            System.out.print(inputLayer[n].output + " ");
        }
        System.out.println("\n");
        */

        /*
        Each hidden layer neuron receives the outputs of neurons of the previous layer.
        (Our neural network has only one hidden layer, so the "previous layer" of the hidden layer
        is always the input layer.)
        Each hidden layer neuron takes these outputs, multiplies each output by the weight
        of the synapse by which it was received, and adds the products. This is done in
        the hiddenNeuronOutput() of Neuron.
        */
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayer[i].calculateOutput(hiddenLayer.length);
        }
        ////////////////////////////////////
        /*
        System.out.println("hiddenLayer :");
        for(int n = 0; n < hiddenLayer.length; n++) {
            System.out.print(hiddenLayer[n].output + " ");
        }
        System.out.println("\n");
         */

        /*
        Calculating the output of our singular output layer neuron.
        */
        double[] analysis = new double[1];
        for(int i = 0; i < outputLayer.length; i++) {
            outputLayer[i].calculateOutput(0);
            analysis[i] = outputLayer[i].output;
        }
        ////////////////////////////////////
        /*
        System.out.println("output :");
        for(int n = 0; n < outputLayer.length; n++) {
            System.out.print(outputLayer[n].output + " ");
        }
        System.out.println("\n");
         */

        //Reset all neurons outputs to 0;
        //Do not want to do this the last forward before a back(), since need to refer to output

        return analysis;
    }

    /*
    The forward() method is forward propagation, and it is used in training and in testing.
    */
    public void forwardTraining(double[] vector, ArrayList<double[]> inputStates,
                                ArrayList<double[]> hiddenStates, ArrayList<double[]> outputStates) {

        double[] inputState = new double[inputLayer.length];
        for (int i = 0; i < inputLayer.length; i++) {
            inputLayer[i].calculateOutput(vector[i]);
            inputState[i] = inputLayer[i].output;
        }
        inputStates.add(inputState);


        /*
        Each hidden layer neuron receives the outputs of neurons of the previous layer.
        (Our neural network has only one hidden layer, so the "previous layer" of the hidden layer
        is always the input layer.)
        Each hidden layer neuron takes these outputs, multiplies each output by the weight
        of the synapse by which it was received, and adds the products. This is done in
        the hiddenNeuronOutput() of Neuron.
        */
        double[] hiddenState = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayer[i].calculateOutput(outputLayer.length);
            hiddenState[i] = hiddenLayer[i].output;
        }
        hiddenStates.add(hiddenState);

        /*
        System.out.println("hiddenLayer :");
        for(int n = 0; n < hiddenLayer.length; n++) {
            System.out.print(hiddenLayer[n].output + " ");
        }
        System.out.println("\n");
         */


        /*
        Calculating the output of our singular output layer neuron.
        */
        double[] outputState = new double[outputLayer.length];
        for(int i = 0; i < outputLayer.length; i++) {
            outputLayer[i].calculateOutput(0);
            outputState[i] = outputLayer[i].output;
        }
        outputStates.add(outputState);

        /*
        System.out.println("output :");
        for(int n = 0; n < outputLayer.length; n++) {
            System.out.print(outputLayer[n].output + " ");
        }
        System.out.println("\n");
         */
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
    public void back(ArrayList<double[]> inputStates, ArrayList<double[]> hiddenStates, ArrayList<double[]> outputStates,
                     ArrayList<double[]> sampleAnswers, double[] weightsUpdate_o, double[] weightsUpdate_h,
                     double[] biasesUpdate_o, double[] biasesUpdate_h,
                     double leftWeights_LR, double rightWeights_LR,
                     double leftBiases_LR, double rightBiases_LR) {

        //ArrayList<Double> dAverageLoss_dNetInput = new ArrayList<>();

        gradient_o(hiddenStates, outputStates, sampleAnswers, weightsUpdate_o, biasesUpdate_o);

        gradient_h( inputLayer, hiddenLayer, outputLayer,
                inputStates, hiddenStates, outputStates, sampleAnswers, weightsUpdate_h, biasesUpdate_h);

        /*
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
        */

        //update weights
        //Between output and hidden2
        for(int outputNeuron = 0; outputNeuron < outputLayer.length; outputNeuron++){
            for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer.length; hiddenNeuron++){
                outputLayer[outputNeuron].getSynapses().get(hiddenNeuron).updateWeight_R(weightsUpdate_o[hiddenNeuron], rightWeights_LR);
            }
        }

        //update output biases
        for(int outputNeuron = 0; outputNeuron < outputLayer.length; outputNeuron++){
            outputLayer[outputNeuron].updateBias_R(biasesUpdate_o[outputNeuron], rightBiases_LR);
        }

        /*
        System.out.println("Between h2 and output:");
        for(int i = 0; i < hiddenLayer2.length; i++){
            System.out.print(outputLayer[i].getSynapses().get(i).weight + " ");
        }
        System.out.println("\n");
        */


        /*
        System.out.println("Between h1 and h2:");
        for(int i = 0; i < hiddenLayer1.length; i++){
            System.out.print(hiddenLayer2[i].getSynapses().get(i).weight + " ");
        }
        System.out.println("\n");
        */

        //Between hidden1 and input
        for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer.length; hiddenNeuron++){
            for(int inputNeuron = 0; inputNeuron < inputLayer.length; inputNeuron++){
                hiddenLayer[hiddenNeuron].getSynapses().get(inputNeuron).updateWeight_L(weightsUpdate_h[inputNeuron], leftWeights_LR);
            }
        }

        for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer.length; hiddenNeuron++){
            hiddenLayer[hiddenNeuron].updateBias_L(biasesUpdate_h[hiddenNeuron], leftBiases_LR);
        }

        /*
        System.out.println("Between input and h1:");
        for(int i = 0; i < inputLayer.length; i++){
            System.out.print(hiddenLayer1[i].getSynapses().get(i).weight + " ");
        }
        System.out.println("\n");
        */

    }

    public void gradient_o(ArrayList<double[]> hiddenStates, ArrayList<double[]> outputStates,
                           ArrayList<double[]> sampleAnswers, double[] weightsUpdate_o, double[] biasesUpdate_o/*, ArrayList<Double> dAverageLoss_dNetInput*/){

        //double[] weightsUpdate =  new double [outputLayer.length * hiddenLayer.length];
        int weightsUpdateIndex = 0;

        //dAverageLoss_dSynapse = dAverageLoss_dOutput * dOutput_dNetInput * dNetInput_dSynapse
        double dAverageLoss_dSynapse = 0.0;
        //double gradient_h_feed = 0.0;

        //want to consider per synapse-weight
        //sum of each of 500 sample's (-2. 0 * (answer - this.output) * derivativeSigmoid(this) * attached leftNeuron.output)
        //therefore need to store output of every neuron for every sample (and zero out if necessary after backprop)

        //then want to consider per bias
        for(int neuron = 0; neuron < outputLayer.length; neuron++) {
            for (int leftNeuron = 0; leftNeuron < hiddenLayer.length; leftNeuron++) {
                for (int sampleState = 0; sampleState < outputStates.size(); sampleState++) {
                    dAverageLoss_dSynapse +=
                            (-2.0 * (sampleAnswers.get(sampleState)[neuron] - outputStates.get(sampleState)[neuron]) *
                                    outputLayer[neuron].derivativeSigmoid(outputStates.get(sampleState)[neuron]) *
                                    hiddenStates.get(sampleState)[leftNeuron]);
                }
                //dAverageLoss_dNetInput.add(gradient_h_feed);
                weightsUpdate_o[weightsUpdateIndex + leftNeuron] = dAverageLoss_dSynapse;
                //gradient_h_feed = 0.0;
                dAverageLoss_dSynapse = 0.0;
            }
            weightsUpdateIndex = hiddenLayer.length;
        }
        //return weightsUpdate;

        double dAverageLoss_dBias = 0.0;

        for(int neuron = 0; neuron < outputLayer.length; neuron++) {
            for (int leftNeuron = 0; leftNeuron < hiddenLayer.length; leftNeuron++) {
                for (int sampleState = 0; sampleState < outputStates.size(); sampleState++) {
                    dAverageLoss_dBias +=
                            (-2.0 * (sampleAnswers.get(sampleState)[neuron] - outputStates.get(sampleState)[neuron]) *
                                    outputLayer[neuron].derivativeSigmoid(outputStates.get(sampleState)[neuron]));
                }
                //dAverageLoss_dNetInput.add(gradient_h_feed);

                //gradient_h_feed = 0.0;

            }
            biasesUpdate_o[neuron] = dAverageLoss_dBias;
            dAverageLoss_dBias = 0.0;
        }
    }

    public void gradient_h(Neuron[] leftLayer, Neuron[] thisLayer, Neuron[] rightLayer,
                           ArrayList<double[]> leftStates, ArrayList<double[]> thisStates, ArrayList<double[]> rightStates,
                           ArrayList<double[]> sampleAnswers, double[] weightsUpdate_h, double[] biasesUpdate_h/*, ArrayList<Double> dAverageLoss_dNetInput*/) {

        //double[] weightsUpdate =  new double [thisLayer.length * leftLayer.length];
        int weightsUpdateIndex = 0;

        //dAverageLoss_dSynapse = dAverageLoss_dOutput * dOutput_dNetInput * dNetInput_dSynapse
        double dAverageLoss_dSynapse = 0;

        for(int neuron = 0; neuron < thisLayer.length; neuron++){
            for(int leftNeuron = 0; leftNeuron < leftLayer.length; leftNeuron++){
                for(int rightNeuron = 0; rightNeuron < rightLayer.length; rightNeuron++){
                    for(int sampleState = 0; sampleState < thisStates.size(); sampleState++){
                        dAverageLoss_dSynapse +=
                                (-2.0 * (((sampleAnswers.get(sampleState)[rightNeuron] - rightStates.get(sampleState)[rightNeuron]) *
                                        rightLayer[rightNeuron].derivativeSigmoid(rightStates.get(sampleState)[rightNeuron])) *
                                        thisLayer[neuron].getSynapse(rightNeuron).weight) *
                                        thisLayer[neuron].derivativeSigmoid(thisStates.get(sampleState)[neuron]) *
                                        leftStates.get(sampleState)[leftNeuron]);
                    }
                }
                weightsUpdate_h[weightsUpdateIndex + leftNeuron] = dAverageLoss_dSynapse;
                dAverageLoss_dSynapse = 0.0;
            }
            weightsUpdateIndex += leftLayer.length;
        }
        //return weightsUpdate;

        double dAverageLoss_dBias = 0;

        for(int neuron = 0; neuron < thisLayer.length; neuron++){
            for(int leftNeuron = 0; leftNeuron < leftLayer.length; leftNeuron++){
                for(int rightNeuron = 0; rightNeuron < rightLayer.length; rightNeuron++){
                    for(int sampleState = 0; sampleState < thisStates.size(); sampleState++){
                        dAverageLoss_dBias +=
                                (-2.0 * (((sampleAnswers.get(sampleState)[rightNeuron] - rightStates.get(sampleState)[rightNeuron]) *
                                        rightLayer[rightNeuron].derivativeSigmoid(rightStates.get(sampleState)[rightNeuron])) *
                                        thisLayer[neuron].getSynapse(rightNeuron).weight) *
                                        thisLayer[neuron].derivativeSigmoid(thisStates.get(sampleState)[neuron]));
                    }
                }
            }
            biasesUpdate_h[neuron] = dAverageLoss_dBias;
            dAverageLoss_dBias = 0.0;
        }
    }

    public void train(){

        double Loss;
        int gradientDescents = 1;
        double LossCalculator = 0.0;
        ArrayList<double[]> inputStates = new ArrayList<>();
        ArrayList<double[]> hiddenStates = new ArrayList<>();
        ArrayList<double[]> outputStates = new ArrayList<>();
        long gradientDescentStart;
        long gradientDescentEnd;
        long gradientDescentTime;
        int[] stochasticIndices = new int[500];
        int[] noDuplicates = new int[500];
        ArrayList<double[]> stochasticSample = new ArrayList<>();
        ArrayList<double[]> sampleAnswers = new ArrayList<>();
        double[] weightsUpdate_o = new double[outputLayer.length * hiddenLayer.length];
        double[] weightsUpdate_h = new double[hiddenLayer.length * inputLayer.length];
        double[] biasesUpdate_o = new double[outputLayer.length];
        double[] biasesUpdate_h = new double[hiddenLayer.length];
        //double rightBiases_LR = 0.00001;
        //double rightWeights_LR = 0.000001;
        //double leftBiases_LR = 0.000001;
        //double leftWeights_LR = 0.009;
        double rightBiases_LR = 0.00001;
        double rightWeights_LR = 0.000001;
        double leftBiases_LR = 0.000009;
        double leftWeights_LR = 0.009;
        boolean learned = false;
        int learnedTrigger = 0;
        int changeTrigger = 0;

        boolean finished = false;
        while (finished == false){

            //Forming a stochastic sample of the training review vectors and their labels
            for(int i = 0; i < stochasticIndices.length; i++){
                boolean duplicate = true;
                while(duplicate == true) {
                    stochasticIndices[i] = new Random().nextInt(25000);
                    boolean noCopies = true;
                    for(int j = 0; j < i; j++) {
                        if(stochasticIndices[i] == noDuplicates[j]){
                            noCopies = false;
                        }
                    }
                    if(noCopies == true){
                        duplicate = false;
                    }
                }
                noDuplicates[i] = stochasticIndices[i];
            }
            /*
            for(int i = 0; i < stochasticIndices.length; i++){
                stochasticSample.add(vectorizer.trainingMatrix.get(i));
                sampleAnswers.add(vectorizer.trainingAnswers.get(i));
            }
            */
            for(int i = 0; i < stochasticIndices.length; i++){
                stochasticSample.add(knawledgeMatrix.get(i));
                sampleAnswers.add(knawledgeAnswersMatrix.get(i));
            }


            /*
            System.out.println("Gradient Descent " + gradientDescents + " 500 random indices:");
            for(int p = 0; p < stochasticIndices.length; p++){
                System.out.print( stochasticIndices[p] + " ");
            }
            System.out.println();
            */

            //Iterate through samples
            gradientDescentStart = System.currentTimeMillis();
            for(int sample = 0; sample < stochasticSample.size(); sample++) {
                forwardTraining(stochasticSample.get(sample), inputStates, hiddenStates, outputStates);

                LossCalculator += Math.pow((outputStates.get(sample)[0] - sampleAnswers.get(sample)[0]),2);

                //System.out.println("Top: " + analysis[0] + ", Bottom: " + analysis[1]);
                //System.out.println("Answers: " + sampleAnswers.get(sample)[0] + " " + sampleAnswers.get(sample)[1]);

                //Zero out all outputs to clean the neural network
                for(int z = 0; z < inputLayer.length; z++){
                    inputLayer[z].output = 0;
                }
                for(int z = 0; z < hiddenLayer.length; z++){
                    hiddenLayer[z].output = 0;
                }
                for(int z = 0; z < outputLayer.length; z++){
                    outputLayer[z].output = 0;
                }
            }

            //Is Loss < 0.1?
            Loss = (LossCalculator / 500.0);

            //Losses.add(Loss);

            back(inputStates, hiddenStates, outputStates, sampleAnswers,
                    weightsUpdate_o, weightsUpdate_h,
                    biasesUpdate_o, biasesUpdate_h,
                    leftWeights_LR, rightWeights_LR,
                    leftBiases_LR, rightBiases_LR);

            gradientDescentEnd = System.currentTimeMillis();
            gradientDescentTime = (gradientDescentEnd - gradientDescentStart) / 1000;

            System.out.println("Gradient Descent " + gradientDescents + " Loss: " + Loss + ", LossCalculator: " + LossCalculator +
                    ", period: " + gradientDescentTime);

            LossCalculator = 0.0;
            inputStates.clear();
            hiddenStates.clear();
            outputStates.clear();
            sampleAnswers.clear();
            stochasticSample.clear();

            //gradientDescentTimes.add(gradientDescentTime);

            gradientDescents++;

            if(Loss <= 0.1 || /*gradientDescents == 100000 ||*/ learned == true/*|| Loss == 1.0*/ || gradientDescents > 100000){
                finished = true;
            }

            if(Loss < 0.20){
                learnedTrigger++;
            }
            if(learnedTrigger >= 1000){
                learned = true;
            }

            if(gradientDescents > 150){
                rightBiases_LR = 0.00001; //0000 1
                rightWeights_LR = 0.00009; //00000 1
                leftBiases_LR = 0.03; //00000 9
                leftWeights_LR = 0.3; // 009
            }

            if(gradientDescents > 250){
                rightBiases_LR = 0.0001; //0000 1
                rightWeights_LR = 0.009; //00000 1
                leftBiases_LR = 0.9; //00000 9
                leftWeights_LR = 3.0; // 00 9
            }


            //if(Loss < 0.73 /*Loss < 0.71 && Loss > 0.61*/){
            //leftWeights_LR = 0.009;

            //rightBiases_LR = 0.00001;
            //rightWeights_LR = 0.0001;
            //leftBiases_LR = 0.001;
            //leftWeights_LR = 10.0;

            //rightBiases_LR = 0.000001;
            //rightWeights_LR = 0.000001;
            //leftBiases_LR = 0.0001;
            //leftWeights_LR = 0.009;
            //}

            //double rightBiases_LR = 0.00001;
            //double rightWeights_LR = 0.000001;
            //double leftBiases_LR = 0.000001;
            //double leftWeights_LR = 0.009;

            /*
            if(Loss < 0.61 && Loss > 0.59){
                leftWeights_LR = 0.0009;
            }

            if(Loss < 0.59){
                leftWeights_LR = 0.002;
            }
             */


            /*
            if(Loss < 0.90){
                changeTrigger++;
            }
            if(changeTrigger >= 3){

                rightBiases_LR = 0.0001;
                rightWeights_LR = 0.00001;
                leftBiases_LR = 0.00001;
                leftWeights_LR = 0.001;
            }
            */
        }



        try {
            File ohFile = new File("src/resources/NewOutputHiddenWeights.txt");
            if (ohFile.createNewFile()) {
                System.out.println("File created: " + ohFile.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter("src/resources/NewOutputHiddenWeights.txt");
            int count = 1;
            for(int i = 0; i < outputLayer.length; i++){
                for(int j = 0; j < hiddenLayer.length; j++){
                    myWriter.write(outputLayer[i].getSynapse(j).weight + " ");
                    if (count % 40 == 0) {
                        myWriter.write(System.getProperty("line.separator"));
                    }
                    count++;
                }
            }
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        try {
            File obFile = new File("src/resources/NewOutputBiases.txt");
            if (obFile.createNewFile()) {
                System.out.println("File created: " + obFile.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter("src/resources/NewOutputBiases.txt");
            for(int i = 0; i < outputLayer.length; i++){
                myWriter.write(outputLayer[i].getBias() + " ");
            }
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        try {
            File hiFile = new File("src/resources/NewHiddenInputWeights.txt");
            if (hiFile.createNewFile()) {
                System.out.println("File created: " + hiFile.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter("src/resources/NewHiddenInputWeights.txt");
            int count = 1;
            for(int i = 0; i < hiddenLayer.length; i++){
                for(int j = 0; j < inputLayer.length; j++){
                    myWriter.write(hiddenLayer[i].getSynapse(j).weight + " ");
                    if (count % 40 == 0) {
                        myWriter.write(System.getProperty("line.separator"));
                    }
                    count++;
                }
            }
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        try {
            File hbFile = new File("src/resources/NewHiddenBiases.txt");
            if (hbFile.createNewFile()) {
                System.out.println("File created: " + hbFile.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter("src/resources/NewHiddenBiases.txt");
            for(int i = 0; i < hiddenLayer.length; i++){
                myWriter.write(hiddenLayer[i].getBias() + " ");
                myWriter.write(System.getProperty("line.separator"));
            }
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        //for(int i = 0; i < gradientDescentTimes.size(); i++) {
        //    System.out.println("Loss: " + Losses.get(i) + ", period: " + gradientDescentTimes.get(i) + " seconds/ ");
        //}
    }

    public void test(){

        //double[] avgLossNumerators = new double[25000];

        int correct = 0;

        //for(int i = 0; i < vectorizer.testMatrix.size(); i++) {
        //double[] analysis = forward(vectorizer.testMatrix.get(i));
        for(int i = 0; i < knawledgeMatrix.size(); i++) {
            double[] analysis = forward(knawledgeMatrix.get(i));
            double[] residuals = new double[1];
            residuals[0] = Math.pow((knawledgeAnswersMatrix.get(i)[0] - analysis[0]),2);
            double loss = residuals[0];
            System.out.println("analysis: " + analysis[0]);
            System.out.println("answer: " + knawledgeAnswersMatrix.get(i)[0]);
            System.out.println("loss: " + loss);

            //avgLossNumerators[i] = loss;

            int choice = 0;
            if(analysis[0] > 0.5){
                choice = 1;
            }

            int answer = 0;
            //if(vectorizer.testAnswers.get(i)[0] < vectorizer.testAnswers.get(i)[1]){
            if(knawledgeAnswersMatrix.get(i)[0] > 0.5){
                answer = 1;
            }


            if(choice == answer){
                correct++;
            }

            for(int z = 0; z < 1; z++){
                analysis[z] = 0.0;
            }
        }

        //double sum = 0;
        //for(int i = 0; i < avgLossNumerators.length; i++){
        //sum += avgLossNumerators[i];
        //}
        //double avgLoss = sum / 25000.00;

        //System.out.println("Average Loss: " + avgLoss);
        System.out.println("Number correct: " + (correct/*vectorizer.testMatrix.size())/250*/));
        System.out.println("Percent correct: " + (double)correct/25000);

    }


    public static void main(String[] args) throws IOException, InterruptedException {

        //Training
        //-----------------------------------------------------------//
        //System.out.println("Training Begins. Teach me.\n");

        //WordVectorizer produces an input, a vector called bagOfWords, for our neural network.
        //vectorizer.readyTrainingData("src/resources/train.csv");

        //System.out.println("Final Dictionary size: " + vectorizer.finalDictionary.size());


        /*
        Initializing our neural network.
        The number of input layer neurons will be wv.bagOfWords.size().
        (This is one input layer neuron for each word of text deemed valuable.)
        There will be one hidden layer of 700 neurons.
        */
        //NeuralNetwork Ultron = new NeuralNetwork(vectorizer.finalDictionary.size(), 10);
        NeuralNetwork Ultron = new NeuralNetwork(397, 30);

        long startTime = System.currentTimeMillis();

        /*
        //Create a reader
        //assign values to an array
        File hiFile = new File("src/resources/HiddenInputWeights.txt");
        FileReader fileReader = new FileReader(hiFile);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        ArrayList<String> linesOfWeights = new ArrayList<>();
        for(String line; (line = bufferedReader.readLine()) != null; ){
            linesOfWeights.add(line);
        }
        ArrayList<String[]> lineVectors = new ArrayList<>();
        for(int i = 0; i < linesOfWeights.size(); i++){
            String[] lineVector;
            lineVector = linesOfWeights.get(i).trim().split("\\s");
            lineVectors.add(lineVector);
        }
        double[] hiWeights = new double[inputLayer.length * hiddenLayer.length];
        for(int i = 0; i < lineVectors.size(); i++){
            for(int j = 0; j < lineVectors.get(i).length; j++){
                hiWeights[(i * 40) + j] = Double.parseDouble(lineVectors.get(i)[j]);
            }
        }
        int HtoIcounter = 0;
        //sets synapse weights from hidden to input
        for(int i = 0; i < hiddenLayer.length; i++){
            for(int j = 0; j < inputLayer.length; j++){
                hiddenLayer[i].getSynapse(j).setWeight(hiWeights[HtoIcounter]);
                //inputLayer[j].getSynapse(i).setWeight(hiWeights[(indexMultiplier * inputLayer.length) + j]);
            }
        }

        /////////////////////////////////////
        //Create a reader
        //assign values to an array
        File ohFile = new File("src/resources/OutputHiddenWeights.txt");
        FileReader ohFileReader = new FileReader(ohFile);
        BufferedReader ohBufferedReader = new BufferedReader(ohFileReader);

        ArrayList<String> ohLinesOfWeights = new ArrayList<>();
        for(String line; (line = ohBufferedReader.readLine()) != null; ){
            ohLinesOfWeights.add(line);
        }
        ArrayList<String[]> ohLineVectors = new ArrayList<>();
        for(int i = 0; i < ohLinesOfWeights.size(); i++){
            String[] lineVector;
            lineVector = ohLinesOfWeights.get(i).trim().split("\\s");
            ohLineVectors.add(lineVector);
        }
        double[] ohWeights = new double[hiddenLayer.length * outputLayer.length];
        for(int i = 0; i < ohLineVectors.size(); i++){
            for(int j = 0; j < ohLineVectors.get(i).length; j++){
                ohWeights[j] = Double.parseDouble(ohLineVectors.get(i)[j]);
            }
        }
        //sets synapse weights from output to hidden
        int OtoHcounter = 0;
        for(int i = 0; i < outputLayer.length; i++){
            for(int j = 0; j < hiddenLayer.length; j++){
                outputLayer[i].getSynapse(j).setWeight(ohWeights[OtoHcounter]);
                //hiddenLayer[j].getSynapse(i).setWeight(hiWeights[(indexMultiplier * hiddenLayer.length) + j]);
                OtoHcounter++;
            }
        }
        //////////////////////////////////////////////////////////
        File hbFile = new File("src/resources/HiddenBiases.txt");
        FileReader hbFileReader = new FileReader(hbFile);
        BufferedReader hbBufferedReader = new BufferedReader(hbFileReader);

        ArrayList<String> hbLinesOfBiases = new ArrayList<>();
        for(String line; (line = hbBufferedReader.readLine()) != null; ){
            hbLinesOfBiases.add(line);
        }
        ArrayList<String[]> hbLineVectors = new ArrayList<>();
        for(int i = 0; i < hbLinesOfBiases.size(); i++){
            String[] lineVector;
            lineVector = hbLinesOfBiases.get(i).trim().split("\\s");
            hbLineVectors.add(lineVector);
        }
        int hBiasesIndex = 0;
        double[] hBiases = new double[hiddenLayer.length];
        for(int i = 0; i < hbLineVectors.size(); i++){
            for(int j = 0; j < hbLineVectors.get(i).length; j++){
                hBiases[hBiasesIndex] = Double.parseDouble(hbLineVectors.get(i)[j]);
                hBiasesIndex++;
            }
        }
        //sets synapse weights from hidden to input
        for(int i = 0; i < hiddenLayer.length; i++){
            hiddenLayer[i].setBias(hBiases[i]);
        }

        File obFile = new File("src/resources/OutputBiases.txt");
        FileReader obFileReader = new FileReader(obFile);
        BufferedReader obBufferedReader = new BufferedReader(obFileReader);

        ArrayList<String> obLinesOfBiases = new ArrayList<>();
        for(String line; (line = obBufferedReader.readLine()) != null; ){
            obLinesOfBiases.add(line);
        }
        ArrayList<String[]> obLineVectors = new ArrayList<>();
        for(int i = 0; i < obLinesOfBiases.size(); i++){
            String[] lineVector;
            lineVector = obLinesOfBiases.get(i).trim().split("\\s");
            obLineVectors.add(lineVector);
        }
        int oBiasesIndex = 0;
        double[] oBiases = new double[outputLayer.length];
        for(int i = 0; i < obLineVectors.size(); i++){
            for(int j = 0; j < obLineVectors.get(i).length; j++){
                oBiases[oBiasesIndex] = Double.parseDouble(obLineVectors.get(i)[j]);
                oBiasesIndex++;
            }
        }
        //sets synapse weights from hidden to input
        for(int i = 0; i < outputLayer.length; i++){
            outputLayer[i].setBias(oBiases[i]);
        }
        */
        ////////////////////////////////////////////////////////
        /*
        System.out.print("output synapses: ");
        for(int i = 0; i < outputLayer.length; i++){
            for(int j = 0; j < hiddenLayer.length; j++){
                System.out.print(outputLayer[i].getSynapse(j).weight + " ");
            }
        }
        System.out.println("\n");

        System.out.print("hidden synapses: ");
        for(int i = 0; i < hiddenLayer.length; i++){
            for(int j = 0; j < outputLayer.length; j++){
                System.out.print(hiddenLayer[i].getSynapse(inputLayer.length + j).weight + " ");
            }
        }
        System.out.println("\n");

        System.out.print("ohWeights: ");
        for(int i = 0; i < ohWeights.length; i++){
            System.out.print(ohWeights[i] + " ");
        }
        */

        /*
        System.out.println("Hidden Biases: ");
        for(int i = 0; i <hiddenLayer.length; i++){
            System.out.println(hiddenLayer[i].getBias());
        }
        System.out.println("Output Biases: ");
        for(int i = 0; i <outputLayer.length; i++){
            System.out.println(outputLayer[i].getBias());
        }
        */


        System.out.println("I wanna learn!\n");


        //Creating feed.txt and feedAnswers.txt
        //////////////////////////
        /*
        try {
            File feed = new File("src/main/resources/feed.txt");
            if (feed.createNewFile()) {
                System.out.println("File created: " + feed.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter("src/main/resources/feed.txt");
            for(int i = 0; i < vectorizer.trainingMatrix.size(); i++){
                for(int j = 0; j < vectorizer.trainingMatrix.get(i).length; j++){
                    myWriter.write(vectorizer.trainingMatrix.get(i)[j] + " ");
                }
            }
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        try {
            File feedAnswers = new File("src/main/resources/feedAnswers.txt");
            if (feedAnswers.createNewFile()) {
                System.out.println("File created: " + feedAnswers.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            FileWriter myWriter = new FileWriter("src/main/resources/feedAnswers.txt");
            for(int i = 0; i < vectorizer.trainingAnswers.size(); i++){
                for(int j = 0; j < vectorizer.trainingAnswers.get(i).length; j++){
                    myWriter.write(vectorizer.trainingAnswers.get(i)[j] + " ");
                }
            }
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        */
        //////////////////////////
        /////////////////////////

        File feed = new File("src/main/resources/feed.txt");
        FileReader feedFileReader = new FileReader(feed);
        BufferedReader feedBufferedReader = new BufferedReader(feedFileReader);

        ArrayList<String> feedLines = new ArrayList<>();
        for(String line; (line = feedBufferedReader.readLine()) != null; ){
            feedLines.add(line);
        }

        ArrayList<String[]> feedLineVectors = new ArrayList<>();
        for(int i = 0; i < feedLines.size(); i++){
            String[] lineVector;
            lineVector = feedLines.get(i).trim().split("\\s");
            feedLineVectors.add(lineVector);
        }
        double[] knawledge = new double[25000 * 397];
        int incrementor = 0;
        for(int i = 0; i < feedLineVectors.size(); i++){
            for(int j = 0; j < feedLineVectors.get(i).length; j++) {
                knawledge[incrementor] = Double.parseDouble(feedLineVectors.get(i)[j]);
                incrementor++;
            }
        }
        int reviewCount = 0;
        while(reviewCount < 25000){
            double[] review = new double[397];
            for(int i = 0; i < 397; i++){
                review[i] = knawledge[(reviewCount * 397) + i];
            }
            knawledgeMatrix.add(review);
            reviewCount++;
        }


        File feedAnswers = new File("src/main/resources/feedAnswers.txt");
        FileReader feedAnswersFileReader = new FileReader(feedAnswers);
        BufferedReader feedAnswersBufferedReader = new BufferedReader(feedAnswersFileReader);

        ArrayList<String> feedAnswersLines = new ArrayList<>();
        for(String line; (line = feedAnswersBufferedReader.readLine()) != null; ){
            feedAnswersLines.add(line);
        }

        ArrayList<String[]> feedAnswersLineVectors = new ArrayList<>();
        for(int i = 0; i < feedAnswersLines.size(); i++){
            String[] lineVector;
            lineVector = feedAnswersLines.get(i).trim().split("\\s");
            feedAnswersLineVectors.add(lineVector);
        }
        double[] knawledgeAnswers = new double[25000];
        int incrementorAnswers = 0;
        for(int i = 0; i < feedAnswersLineVectors.size(); i++){
            for(int j = 0; j < feedAnswersLineVectors.get(i).length; j++) {
                knawledgeAnswers[incrementorAnswers] = Double.parseDouble(feedAnswersLineVectors.get(i)[j]);
                incrementorAnswers++;
            }
        }
        reviewCount = 0;
        while(reviewCount < 25000){
            double[] review = new double[1];
            for(int i = 0; i < 1; i++){
                review[i] = knawledgeAnswers[reviewCount + i];
            }
            knawledgeAnswersMatrix.add(review);
            reviewCount++;
        }

        ////////////////////////////

        //We train the network for 5 epochs.

        ///////////////////////////////////////////////////////////
        Ultron.train();


        System.out.println("I am ready!");
        //-----------------------------------------------------------//


        //Testing
        //-----------------------------------------------------------//
        System.out.println("I have surpassed you, humans. Gather the test.");



        //WordVectorizer takes out test data as the input for our neural network.
        //vectorizer.readyTestData("src/resources/test.csv");

        //Ultron.test();

        /*
        int testReview = 1;
        for(int i = 0; i < vectorizer.testMatrix.size(); i++){
            System.out.println("test review " + testReview + ": ");
            for(int j = 0; j < vectorizer.testMatrix.get(i).length; j++){
                System.out.print(vectorizer.testMatrix.get(i)[j] + " ");
            }
            System.out.println("\n");
            testReview++;
        }
        */

        long endTime = System.currentTimeMillis();

        long time =  (endTime - startTime)/(1000 * 60);

        System.out.println("Process took " + time + " minutes.");



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