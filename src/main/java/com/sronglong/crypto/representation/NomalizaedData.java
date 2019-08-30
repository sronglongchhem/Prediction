package com.sronglong.crypto.representation;

import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.List;

public class NomalizaedData {

    public static int VECTOR_SIZE = 5;
    /** minimal values of each feature in stock dataset */
    private double[] minArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] maxArray = new double[VECTOR_SIZE];

    /** minimal values of each feature in stock dataset */
    private double[] meanArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] stvArray = new double[VECTOR_SIZE];

    private double[]median = new double[VECTOR_SIZE];

    private List<StockData> stockData;

//    public static void main (String[] args) throws IOException {
//
//        String BTC = new ClassPathResource("BTC_daily__training.csv").getFile().getAbsolutePath();
//        String ETH = new ClassPathResource("ETH_daily__training.csv").getFile().getAbsolutePath();
//
//        readData(BTC);
//    }
//


    public void readData(String filename) throws IOException{

        for (int i = 0; i < maxArray.length; i++) { // initialize max and min arrays
            maxArray[i] = Double.MIN_VALUE;
            minArray[i] = Double.MAX_VALUE;
        }
        List<String[]> list = new CSVReader(new FileReader(filename)).readAll(); // load all elements in a list
        int t = 0;
        for (String[] arr : list) {
            if (!arr[2].equals("open")){
                double[] nums = new double[VECTOR_SIZE];
                for (int i = 0; i < arr.length -(arr.length - VECTOR_SIZE); i++) {
                    nums[i] = Double.valueOf(arr[i + 2]);
                    if (nums[i] > maxArray[i]) maxArray[i] = nums[i];
                    if (nums[i] < minArray[i]) minArray[i] = nums[i];
                }
                stockData.add(new StockData(arr[0], arr[1], nums[0], nums[1], nums[2], nums[3], 0,0));

//                    t++;
//                    if (t > 1000){
//                        break;
//                    }
            }

        }
    }



}
