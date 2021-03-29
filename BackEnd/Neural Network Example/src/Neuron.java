import java.util.*;


public class Neuron {

    //The leakinessParameter variable multiplies the input value in the ReLU function
    static double leakinessParameter = 0.08;
    //The output of any neuron after processing its input(s)
    double output;
    //Synapses connect neurons from one layer of neurons to the next
    ArrayList<Synapses> synapses;

    /*
    Constructor.
    This form of constructor is for neurons that will be in the first layer (input layer)
    of neurons. Such neurons would not have inputs from previous neurons.
    */
    public Neuron() {

        output =0;

        synapses = new ArrayList<Synapses>();
    }
    /*
    Constructor.
    This form of constructor is for neurons that will be in all layers but the first layer.
    */
    public Neuron(double output) {

        this.output = output;

        synapses = new ArrayList<Synapses>();
    }

    /*
    The neuronOutput() method is used by a neuron to multiply incoming inputs
    with respective weights. The resulting products are summed for the neuron's single output.
    */
    public void neuronOutput() {

        double sum =0;

        for(int i=0; i < synapses.size(); i++) {

            Synapses c = synapses.get(i);

            /*
            Initializing one neuron as one end (source) for a synapse connection and
            one neuron as the other end (destination).
             */
            Neuron source = c.source;
            Neuron destination = c.destination;

            /*
            (Not for the input layer (first layer) of neurons (that would not have neurons
            sending information to them) (Hence if-statement to make sure neuron is a destination
            of a synapse between neurons)
            This produces the output of this method (and this neuron).
            */
            if(this == destination) {

                sum+=(c.weight * source.output);


            }

        }

        /*
        A neuron could have an output much larger than the output of another neuron. We do not want a neuron to
        overpower others to a point that the other neurons are without influence whatsoever.
        This function forces our neuron's output to be within a -1 to 1 range.
        */
        output = tanh_func(sum);
    }


    /*
    The hiddenNeuronOutput() method is like the neuronOutput method above, but for
    neurons that are in layers between the initial and final layer.
    The only difference is the treatment of the output before it is returned. The exact point
    of difference is indicated by comment below.
    */
    public void hiddenNeuronOutput() {

        double sum =0;

        for(int i=0; i < synapses.size(); i++) {

            Synapses c = synapses.get(i);

            Neuron source = c.source;
            Neuron destination = c.destination;

            if(this == destination) {

                sum+=(c.weight * source.output);


            }

        }

        //This is the different treatment of the output.
        output = ReLU(sum);
    }

    /*
    This is called by neuronOutput().
    This forces a neuron's output to be between -1 and 1.
    */
    double tanh_func(double y) {

        return (( Math.exp(y) - Math.exp(-y))/( Math.exp(y) + Math.exp(-y)));
    }

    /*
    This is called by hiddenNeuronOutput().
    ReLU stands for 'rectified linear unit'.
    */
    static double ReLU(double y) {

        return y >=0 ? y : y* leakinessParameter;
    }

    //This is the derivative of the RelU for backpropagation.
    static double derivativeReLU(double y) {

        double derivative = y > 0 ? 1.0 :0.01;

        return derivative;
    }


    //getSynapses() returns the array of synapses between neurons
    public ArrayList<Synapses> getSynapses() {

        return synapses;

    }

    //addSynapses() adds a synapse.
    public void addSynapses(Synapses synapse) {

        synapses.add(synapse);

    }

}