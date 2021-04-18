import java.util.Random;


//A synapse connects one neuron to another neuron
public class Synapse {

    Neuron left;

    Neuron right;

    double weight;


    /*
    In this constructor, the weight (importance, influence) of a synapse is
    initialized to a random number so that we begin from somewhere/no where
    other than 0 or all synapses being equal (similar to all being 0).
    */
    Synapse(Neuron left, Neuron right){

        this.left = left;

        this.right = right;

        weight = (new Random().nextDouble() * 0.5 + 0.1);
    }

    Synapse(Neuron left, Neuron right, double weight){

        this.left = left;

        this.right = right;

        this.weight = weight;
    }

    /*
    We use gradient-descent in order for the weight of the synapses to be adjusted
    during our neural network's training. This adjustment of the weights of synapses
    is "learning".
    Each synapse weight updates according to the parameter argument "update."
    "update" is the product of error and current input.
     */
    public /*double*/ void updateWeight(double gradientDescent, double learningRate) {

        /*return weight = weight - (learningRate * gradientDescent);*/
        this.weight = weight - (learningRate * gradientDescent);

    }

    public double getWeight() {

        return weight;
    }

}