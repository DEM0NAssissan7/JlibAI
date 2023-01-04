/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.nobody.libai;

/**
 *
 * @author mawl
 */
public class Util {
    final static double max_random = 1;
    public static double sigmoid(double input){
        return 1 / (1 + Math.pow(Math.E, -input));
    }
    public static double relu(double input){
        if(input > 0)
            return input;
        else
            return 0;
    }
    public static double random(){
        return ((Math.random() - 0.5) * 2) * max_random;
    }
    public static double square(double num){
        return num * num;
    }
    public static double squish(double input){
        return sigmoid(input);
//        return relu(input);
//        return input;
    }
    public static double inverse(double input){
        return input * (1 - input);
//        return input;
    }
}
