package edu.ucmo.spring_bare_bones;

import java.util.ArrayList;
import java.util.Random;


public class Neuron {

    double bias;
    //The leakinessParameter variable multiplies the input value in the ReLU function
    //static double leakinessParameter = 0.08;
    //The output of any neuron after processing its input(s)
    double output;
    //Synapses connect neurons from one layer of neurons to the next
    ArrayList<Synapse> synapses;

    /*
    Constructor.
    This form of constructor is for neurons that will be in the first layer (input layer)
    of neurons. Such neurons would not have inputs from previous neurons.
    */
    public Neuron() {

        bias = new Random().nextDouble() * 0.5 + 0.05;
        output = 0;
        synapses = new ArrayList<Synapse>();
    }

    public Neuron(double bias) {

        this.bias = bias;
        output = 0;
        synapses = new ArrayList<Synapse>();
    }

    //For input layer only
    public void calculateOutput(double feedInput) {
        //output = ReLU(feedInput);
        output = sigmoid(feedInput);
        //output = Math.tanh(feedInput);
    }

    /*
    The neuronOutput() method is used by a neuron to multiply incoming inputs
    with respective weights. The resulting products are summed for the neuron's single output.
    */
    public void calculateOutput(int rightLayerSize) {

        double netInput = 0;

        for(int i = 0; i < synapses.size() - rightLayerSize; i++) {

            Synapse synapse = synapses.get(i); //Per synapse connected to this neuron...

            /*
            Naming previous layer's "neuron i" as one end (source) for synapse and
            this neuron as one end (destination).
            */
            Neuron left = synapse.left;

            /*
            (Not for the input layer (first layer) of neurons (that would not have neurons
            sending information to them) (Hence if-statement to make sure neuron is a destination
            of a synapse between neurons)
            This produces the output of this method (and this neuron).
            */
            //if(this == right) {

            netInput += (synapse.weight * left.output);

            //}

        }
        netInput += bias;

        /*
        A neuron could have an output much larger than the output of another neuron. We do not want a neuron to
        overpower others to a point that the other neurons are without influence whatsoever.
        This function forces our neuron's output to be within a -1 to 1 range.
        */
        //output = ReLU(netInput);
        output = sigmoid(netInput);
        //output = Math.tanh(netInput);
    }

    /*
    This is called by hiddenNeuronOutput().
    ReLU stands for 'rectified linear unit'.
    */
    /*
    static double ReLU(double netInput) {

        if(netInput > 0.0) {
            return netInput;
        }
        else{
            return 0.0;
        }
    }
    */
    //This is the derivative of the RelU for backpropagation.
    /*
    static double derivativeReLU(double number) {

        if(number > 0 ){
            return 1;
        }
        else{
            return 0.0;
        }
    }
    */

    public double sigmoid(double netIntput){
        double output = 0.0;
        output = 1/(1 + Math.exp(-1.0 * netIntput));
        return output;
    }

    public double derivativeSigmoid(double output){
        return (output * (1 - output));
    }

    public double derivativeTanh(){
        return (1 - Math.pow((Math.tanh(this.output)),2));
    }


    //getSynapses() returns the array of synapses between neurons
    public ArrayList<Synapse> getSynapses() {

        return synapses;

    }


    public Synapse getSynapse(int neuron) {

        return synapses.get(neuron);

    }

    //addSynapses() adds a synapse.
    public void addSynapse(Synapse synapse) {

        synapses.add(synapse);

    }

    public void updateBias_L(double bias, double leftBiases_LR){
        this.bias -= (leftBiases_LR * bias);
    }

    public void updateBias_R(double bias, double rightBiases_LR){
        this.bias -= (rightBiases_LR * bias);
    }

    public double getBias(){
        return this.bias;
    }

    public void setBias(double bias){
        this.bias = bias;
    }

}