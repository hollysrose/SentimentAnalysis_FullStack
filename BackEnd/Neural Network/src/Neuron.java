import java.util.ArrayList;
import java.util.Random;


public class Neuron {

    double bias;
    //The leakinessParameter variable multiplies the input value in the ReLU function
    //static double leakinessParameter = 0.08;
    //The output of any neuron after processing its input(s)
    double output;
    //Synapses connect neurons from one layer of neurons to the next
    ArrayList<Synapse> leftSynapses;
    ArrayList<Synapse> rightSynapses;

    /*
    Constructor.
    This form of constructor is for neurons that will be in the first layer (input layer)
    of neurons. Such neurons would not have inputs from previous neurons.
    */
    public Neuron() {

        bias = new Random().nextDouble() * 0.5;
        output = 0;
        leftSynapses = new ArrayList<Synapse>();
        rightSynapses = new ArrayList<Synapse>();
    }

    public Neuron(double bias) {

        this.bias = bias;
        output = 0;
        leftSynapses = new ArrayList<Synapse>();
        rightSynapses = new ArrayList<Synapse>();
    }

    //For input layer only
    public void calculateOutput(double feedInput) {
        output = ReLU(feedInput);
    }

    /*
    The neuronOutput() method is used by a neuron to multiply incoming inputs
    with respective weights. The resulting products are summed for the neuron's single output.
    */
    public void calculateOutput() {

        double netInput = 0;

        for(int i = 0; i < leftSynapses.size(); i++) {

            Synapse synapse = leftSynapses.get(i); //Per synapse connected to this neuron...

            /*
            Naming previous layer's "neuron i" as one end (source) for synapse and
            this neuron as one end (destination).
            */
            Neuron left = synapse.left;
            Neuron right = synapse.right;

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
        output = ReLU(netInput);
    }

    /*
    This is called by hiddenNeuronOutput().
    ReLU stands for 'rectified linear unit'.
    */
    static double ReLU(double netInput) {

        if(netInput > 0.0) {
            return netInput;
        }
        else{
            return 0.0;
        }
    }

    //This is the derivative of the RelU for backpropagation.
    static double derivativeReLU(double number) {

        if(number > 0 ){
            return 1;
        }
        else{
            return 0.1;
        }
    }


    //getSynapses() returns the array of synapses between neurons
    public ArrayList<Synapse> getLeftSynapses() {

        return leftSynapses;

    }

    public ArrayList<Synapse> getRightSynapses() {

        return rightSynapses;

    }

    /*
    public Synapse getSynapse(int neuron) {

        return synapses.get(neuron);

    }
    */

    //addSynapses() adds a synapse.
    public void addLeftSynapse(Synapse synapse) {

        leftSynapses.add(synapse);

    }

    public void addRightSynapse(Synapse synapse) {

        rightSynapses.add(synapse);

    }

}