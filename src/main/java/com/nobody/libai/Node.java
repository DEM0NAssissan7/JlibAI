/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.nobody.libai;

/**
 *
 * @author mawl
 */
public class Node {
    double value = 0;
    double gradient = 0;
    
    double bias = 0;
    double bias_change = 0;
    
    double[] weights;
    double[] changes;
    
    public Node(int previous_layer_size){
        weights = new double[previous_layer_size];
        changes = new double[previous_layer_size];
        for(int i = 0; i < previous_layer_size; i++){
            weights[i] = Util.random();
            changes[i] = 0;
        }
    }
}
