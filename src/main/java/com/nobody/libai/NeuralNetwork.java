/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.nobody.libai;

/**
 *
 * @author mawl
 */
public class NeuralNetwork {
    static Node[][] layers;
    static int[] layout;
    static int epoch;
    static double learning_rate = 0;
    static double cost = 0;
    static double average_cost = 0;
    static long runs = 0;
    
    public NeuralNetwork(int[] layout, int epoch, double learning_rate){
        
        //Init variables
        this.layout = layout;
        this.epoch = epoch;
        this.learning_rate = learning_rate;
        
        //Initialize layers;
        layers = new Node[layout.length][];
        for(int i = 0; i < layout.length; i++){
            layers[i] = new Node[layout[i]];
            for(int j = 0; j < layout[i]; j++){
                if(i > 0)
                    layers[i][j] = new Node(layout[i - 1]);
                else
                    layers[i][j] = new Node(0);
            }
        }
    }
    public static void run(double[] inputs){
        //Set first layer to inputs
        for(int i = 0; i < inputs.length; i++)
            layers[0][i].value = inputs[i];
        
        //Calculate network output
        for (int i = 1; i < layers.length; i++) {
            for (int j = 0; j < layers[i].length; j++) {
                Node node = layers[i][j];
                node.value = 0;
                for(int k = 0; k < layers[i - 1].length; k++)
                    node.value += layers[i - 1][k].value * node.weights[k];
                node.value += node.bias;
                node.value = Util.squish(node.value);
            }
        }
    }
    public static void get_cost(double[] targets){
        //Calculate gradients and changes for the output layer and the layer before it
        cost = 0;
        for(int i = 0; i < layers[layers.length - 1].length; i++)
            cost += Util.square(layers[layers.length - 1][i].value - targets[i]);
        average_cost = (cost + epoch * average_cost) / (epoch + 1);
    }
    public static void learn(double[] targets){
        /*  In backpropogation, we begin from the last layer and keep moving up until we reach the end of the network.
            Using backpropogation, we can tweak all the weights and biases so that the cost of the network will go down.
        
            This is a thing in calculus called 'gradient descent' where, in order to find the lowest point of a function, you
            find the derivative and head in the negative direction. Sometimes, you can end up in a local minimum and not actually where you want to be.
            
            To solve this, we use something called an epoch. The epoch basically caches all of the changes to the weights until it finishes.
            When it finishes, it applies all of the accumilated weight chanages. This means that results are not one-sided and multiple different test trails
            can contribute to the general direction of the descent of the neural network.
        
            (Of course, we want the cost to be as low as possible, but a good rule of thumb is 1.0E-4 [0.0001])
        */
        //gradients and changes for the output layer and the layer before it
        for(int i = 0; i < layers[layers.length - 1].length; i++){
            Node node = layers[layers.length - 1][i];
            node.gradient = (2 * (node.value - targets[i])) * (node.value * (1 - node.value));
            node.bias_change -= learning_rate * node.gradient;
            for(int j = 0; j < node.weights.length; j++)
                node.changes[j] -= learning_rate * (node.gradient * layers[layers.length - 2][j].value);
        }
        
        //Calculate the cost of the network
        get_cost(targets);
        
        //Update the rest of the layers
        for(int i = layers.length - 2; i > 0; i--){
            for(int j = 0; j < layers[i].length; j++){
                Node node = layers[i][j];
                node.gradient = sum_gradient(j, i + 1) * (node.value * (1 - node.value));
                node.bias_change -= learning_rate * node.gradient;
                for(int k = 0; k < node.weights.length; k++)
                    node.changes[k] -= learning_rate * (node.gradient * layers[i - 1][k].value);
            }
        }
        
        //Apply weight changes after an epoch
        if(runs % epoch == epoch - 1){
            for (Node[] layer : layers) {
                for (Node node : layer) {
                    //Apply bias change
                    node.bias += node.bias_change;
                    node.bias_change = 0;
                    for (int k = 0; k < node.weights.length; k++) {
                        //Apply weight changes
                        node.weights[k] += node.changes[k];
                        node.changes[k] = 0;
                    }
                }
            }
        }
        
        runs++;
    }
    public static double sum_gradient(int node_index, int layer_index){
        double result = 0;
        for (Node node : layers[layer_index])
            result += node.weights[node_index] * node.gradient;
        return result;
    }
    public static void train(double[] inputs, double[] targets){
        run(inputs);
        learn(targets);
    }
    public static void print_output(){
        for(int i = 0; i < layers[layers.length - 1].length; i++)
            System.out.println(layers[layers.length - 1][i].value);
    }
    public static void print_network(){
        System.out.println("\nFull network print:\n");
        for(int i = 0; i < layers.length; i++){
            System.out.println("-------------------\nLayer " + i);
            for(int j = 0; j < layers[i].length; j++){
                Node node = layers[i][j];
                System.out.print("\nNode " + j + "\n ++ ");
                for(int k = 0; k < node.weights.length; k++)
                    System.out.print(node.weights[k] + " ++ ");
                System.out.print("\n");
            }
        }
    }
}
