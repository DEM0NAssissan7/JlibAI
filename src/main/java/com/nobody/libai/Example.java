/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.nobody.libai;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author mawl
 */
public class Example {

    static int[] layout = {2, 2, 1};
    static int epoch = 4;
    static double learning_rate = 0.07;
    static NeuralNetwork network = new NeuralNetwork(layout, epoch, learning_rate);
    
    static int trials = 0;
    static double[] input = {0, 0};
    static double[] target = {0};
    static double[][] t_inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {0, 0}
    };
    static double[][] t_targets = {
        {0},
        {1},
        {1},
        {0}
    };
    
    public static void set_io(){
        int index = trials % 4;
        input = t_inputs[index];
        target = t_targets[index];
        trials++;
    }
    public static void train(){
            set_io();
            network.train(input, target);
    }
    static void test(){
        for(int i = 0; i < 4; i++){
            set_io();
            network.run(input);
            network.get_cost(target);
            System.out.println(network.layers[network.layers.length - 1][0].value + " (" + (t_targets[(trials - 1) % 4][0] + ")"));
        }
    }
    public static void main(String[] args) {
        /* This is a simple test of the neural network to see if it can solve the pattern for the XOR function */  
        
        //Run an initial test
        System.out.println("Before training:");
        test();

        //Train the algorithm
        for(int i = 0; i < 10000000; i++)
            train();

        //See the result
        System.out.println("\nTraining complete!");
        System.out.println("Average cost: " + network.average_cost);
        test();
    }
}
