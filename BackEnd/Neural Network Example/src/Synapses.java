import java.util.Random;


//A synapse connects one neuron to another neuron
public class Synapses {

    Neuron source;

    Neuron destination;

    double weight = 0;


    /*
    In this constructor, the weight (importance, influence) of a synapse is
    initialized to a random number so that we begin from somewhere/no where
    other than 0 or all synapses being equal (similar to all being 0).
    */
    Synapses(Neuron source, Neuron destination){

        this.source = source;

        this.destination = destination;

        weight = (new Random().nextDouble()-0.45);
    }

    /*
    We use gradient-descent in order for the weight of the synapses to be adjusted
    during our neural network's training. This adjustment of the weights of synapses
    is "learning".
    Each synapse weight updates according to the parameter argument "update."
    "update" is the product of error and current input.
     */
    public double updateWeight(double update) {

        return weight-=update;

    }

    public double getWeight() {

        return weight;
    }


}