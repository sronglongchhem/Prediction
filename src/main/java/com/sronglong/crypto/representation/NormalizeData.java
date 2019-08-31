package com.sronglong.crypto.representation;

import com.opencsv.CSVReader;
import com.sronglong.crypto.utils.CsvWriterExamples;
import com.sronglong.crypto.utils.EvaluationMatrix;
import com.sronglong.crypto.utils.Helpers;
import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.rank.Median;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by zhanghao on 26/7/17.
 * Modifired by Sronglong
 * @author ZHANG HAO
 */
public class NormalizeData {
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

    private NormalizeType normalizeType;

    private List<StockData> stockData;

//    public static void main (String[] args) throws IOException {
//
//        String BTC = new ClassPathResource("BTC_daily__training.csv").getFile().getAbsolutePath();
//        String ETH = new ClassPathResource("ETH_daily__training.csv").getFile().getAbsolutePath();
//
//        readData(BTC);
//    }
//

//    public NormalizeType () {return }

    public NormalizeData(String filename,String name) throws IOException {
        this.stockData = readData(filename);
        calMeanSTD(stockData);
        this.normalizeType = normalizeType;
//        MINMAX , DECIMAL_SCALING,Z_SCORE,MEDIAN_NOR,SIGMOID_NOR, TANH_EST;
       String localtion =  writeFile(toStringList(stockData,filename),name);
       System.out.println(localtion);

    }

    public List<String[]> toStringList(List<StockData> stockData, String filename) {
        List<String[]> list = new ArrayList<>();
//        list.add(new String[]{normalizeType.toString(), filename});
        list.add(new String[]{"REAL","MINMAX", "DECIMAL_SCALING","Z_SCORE","MEDIAN_NOR","SIGMOID_NOR","TANH_EST"});
        for (int i = 0; i< stockData.size() ; i++){
            StockData data = stockData.get(i);
            list.add(new String[]{
                    String.valueOf(data.getClose()),
                    String.valueOf(calculateValue(data.getClose(),1,NormalizeType.MINMAX)),
                    String.valueOf(calculateValue(data.getClose(),1,NormalizeType.DECIMAL_SCALING)),
                    String.valueOf(calculateValue(data.getClose(),1,NormalizeType.Z_SCORE)),
                    String.valueOf(calculateValue(data.getClose(),1,NormalizeType.MEDIAN_NOR)),
                    String.valueOf(calculateValue(data.getClose(),1,NormalizeType.SIGMOID_NOR)),
                    String.valueOf(calculateValue(data.getClose(),1,NormalizeType.TANH_EST))
            });
        }
        return list;
    }

    public static String writeFile(List<String[]> stockData,String name){
        Path path = null;
        try {
            path = Helpers.fileOutOnePath(name);
        } catch (Exception ex) {
            Helpers.err(ex);
        }

        return CsvWriterExamples.csvWriterAll(stockData,path);
    }


    public List<StockData> readData(String filename) throws IOException{

        List<StockData> stockDataList = new ArrayList<>();

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
                stockDataList.add(new StockData(arr[0], arr[1], nums[0], nums[1], nums[2], nums[3], 0,0));

//                    t++;
//                    if (t > 1000){
//                        break;
//                    }
            }

        }
        return stockDataList;
    }

    private void calculateParameters(double[] input, int index) {

        meanArray[index] = new Mean().evaluate(input); // mean
        stvArray[index] = new StandardDeviation().evaluate(input);
        median[index] =  new Median().evaluate(input);

    }

    /**
     * Calculate Tanh estimator mean and deviation values for dataset values
     * @param dataset List of TrainDataItems
     */
    public void calMeanSTD(List<StockData> dataset) {

        final int num_el_dataitem = 4; //*2;
        double[] arrOpen = new double[dataset.size()];
        double[] arrClose = new double[dataset.size()];
        double[] arrLow = new double[dataset.size()];
        double[] arrHigh = new double[dataset.size()];

        // put everything in one array so proper parameters can be calculated for the whole set
//        Iterator<TrainDataItem> it = dataset.iterator();

        for (int i = 0; i < dataset.size(); i++) {
            StockData stock = dataset.get(i);
            arrOpen[i] = stock.getOpen();
            arrClose[i] = stock.getClose();
            arrLow[i] = stock.getLow();
            arrHigh[i] = stock.getHigh();
        }

        calculateParameters(arrOpen,0);
        calculateParameters(arrClose,1);
        calculateParameters(arrLow,2);
        calculateParameters(arrHigh,3);
    }




    private double calculateValue(double value, int index, NormalizeType normalizeType){
        if (index == 4){
            return  0;
        }
        switch (normalizeType) {
            case MINMAX: return minMaxNormalizer(value,index);
            case DECIMAL_SCALING: return decimalScalingNormalization(value,index);
            case Z_SCORE: return zScore(value,index);
            case MEDIAN_NOR: return medianNormalization(value,index);
            case SIGMOID_NOR: return sigmoidNormalization(value,index);
            case TANH_EST: return tanhestimators(value,index);
            default: throw new NoSuchElementException();
        }
    }

    private double decimalScalingNormalization(double value, int index){
        int length = (int) Math.log10(maxArray[index]) + 1;
        double v1 = value/Math.pow(10, length);
//        System.out.println("Decimal Scalled value of "+value+"="+v1);
//        System.out.println("denorm "+v1+"= "+(v1 *  Math.pow(10, length) ));
        return  v1;
    }

    public double dedecimalScalingNormalization(double value,int index){
        //   System.out.println("df");
        int length = (int) Math.log10(maxArray[index]) + 1;
        return  (value *  Math.pow(10, length) );
    }

    private double minMaxNormalizer(double value, int index){
        double V1=   (value - minArray[index]) / (maxArray[index] - minArray[index]);

//        System.out.println("minMaxNormalizer value of "+value+"="+V1);
//        System.out.println("denorm "+V1+"= "+(V1 * (maxArray[index]- minArray[index]) + minArray[index] ));

        return V1;
    }


    public double denormalMinMAx(double value,int index){
        //   System.out.println("df");
        return  (value * (maxArray[index]- minArray[index]) + minArray[index] );
    }


    private double zScore(double value, int index){
        double V1=(value- meanArray[index])/stvArray[index];
//            System.out.print(V1+",");

        //  System.out.println("zScore value of "+value+"="+V1);
        //  System.out.println("denorm "+V1+"= "+(V1 * (stvArray[index]) + meanArray[index] ));

        return V1;
    }

    public double dezScore(double value,int index){
        //   System.out.println("df");
        return  value * (stvArray[index]) + meanArray[index] ;
    }


    private double medianNormalization(double value, int index){
        double V1= value / median[index];
        //     System.out.println("medianNormalization value of "+value+"="+V1);
        //    System.out.println("denorm "+V1+"= "+(V1 * median[index] ));
        return V1;
    }

    public double demedianNormalization(double value,int index){
        //   System.out.println("df");
        return  value * median[index];
    }


    private double sigmoidNormalization(double value, int index){
        double V1 = 1/(1+Math.exp(-value));
        //   System.out.println("medianNormalization value of "+value+"="+V1);
        //    System.out.println("denorm "+V1+"= "+(V1 * median[index] ));
        return V1;
    }

    public double desigmoidNormalization(double value, int index){
        return value; // cannot reverse
    }

    private double tanhestimators(double value, int index){
        double V1 = 0.5 * (  Math.tanh(0.01 * (value - minArray[index]) /  maxArray[index] ) + 1 );
        return V1;

    }

    public double detanhestimators(double value,int index){
        //   System.out.println("df");
        return EvaluationMatrix.deTanh(value,stvArray[index],meanArray[index]);
    }


}
