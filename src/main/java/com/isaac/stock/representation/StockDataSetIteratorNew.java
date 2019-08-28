package com.isaac.stock.representation;

import breeze.linalg.max;
import com.google.common.collect.ImmutableMap;
import com.isaac.stock.utils.EvaluationMatrix;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math3.analysis.function.Atanh;
import org.apache.commons.math3.analysis.function.Tanh;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

@SuppressWarnings("serial")
public class StockDataSetIteratorNew implements DataSetIterator {
    /** category and its index */
    private final Map<PriceCategory, Integer> featureMapIndex = ImmutableMap.of(PriceCategory.OPEN, 0, PriceCategory.CLOSE, 1,
            PriceCategory.LOW, 2, PriceCategory.HIGH, 3, PriceCategory.VOLUME, 4);

    private final int VECTOR_SIZE = 5; // number of features for a stock data
    private int miniBatchSize; // mini-batch size
    private int exampleLength = 22; // default 22, say, 22 working days per month
    private int predictLength = 1; // default 1, say, one day ahead prediction

    /** minimal values of each feature in stock dataset */
    private double[] minArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] maxArray = new double[VECTOR_SIZE];

    /** minimal values of each feature in stock dataset */
    private double[] meanArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] stvArray = new double[VECTOR_SIZE];

    private double[]median = new double[VECTOR_SIZE];

    /** feature to be selected as a training target */
    private PriceCategory category;

    private NormalizeType normalizeType;

    /** mini-batch offset */
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    /** stock dataset for training */
    private List<StockData> train;


    /** adjusted stock dataset for testing */
    private List<Pair<INDArray, INDArray>> test;

    public StockDataSetIteratorNew(String filename, int miniBatchSize, int exampleLength, double splitRatio, PriceCategory category, NormalizeType normalizeType) {
        List<StockData> stockDataList = readStockDataFromFile(filename);
        this.calMeanSTD(stockDataList);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(stockDataList.size() * splitRatio);
        this.normalizeType = normalizeType;
        train = stockDataList.subList(0, split);
        test = generateTestDataSet(stockDataList.subList(split, stockDataList.size()));
//        initializeOffsets();
    }

    public void spliteTrainandValidate(double splitRatio, boolean isFirst){
        List<StockData> beforeSplit = train;
        int split = (int) Math.round(this.train.size() * splitRatio);
        if (isFirst){
            train = beforeSplit.subList(0,split);
        }else {
            train = train.subList(split,train.size());
//            test = generateTestDataSet(beforeSplit.subList(split, beforeSplit.size()));
        }

        initializeOffsets();

    }


    /** initialize the mini-batch offsets */
    private void initializeOffsets () {
        exampleStartOffsets.clear();
        int window = exampleLength + predictLength;
        for (int i = 0; i < train.size() - window; i++) { exampleStartOffsets.add(i); }
    }

    public List<Pair<INDArray, INDArray>> getTestDataSet() { return test; }

    public double[] getMaxArray() { return maxArray; }

    public double[] getMinArray() { return minArray; }

    public double getMaxNum (PriceCategory category) { return maxArray[featureMapIndex.get(category)]; }

    public double getMinNum (PriceCategory category) { return minArray[featureMapIndex.get(category)]; }

    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label;

        label = Nd4j.create(new int[] {actualMiniBatchSize, predictLength, exampleLength}, 'f');
        
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            StockData curData = train.get(startIdx);
            StockData nextData;
            for (int i = startIdx; i < endIdx; i++) {
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, calculateValue(curData.getOpen(),0));
                input.putScalar(new int[] {index, 1, c}, calculateValue(curData.getClose(),1));
                input.putScalar(new int[] {index, 2, c}, calculateValue(curData.getLow(),2));
                input.putScalar(new int[] {index, 3, c}, calculateValue(curData.getHigh(),3));
                if (VECTOR_SIZE == 5){
                    input.putScalar(new int[] {index, 4, c}, calculateValue(curData.getVolume(),4));
                }


                nextData = train.get(i + 1);
                label.putScalar(new int[]{index, 0, c}, feedLabel(nextData));

                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    private double feedLabel(StockData data) {
        double value;
        switch (category) {
            case OPEN: value = calculateValue(data.getOpen(),0); break;
            case CLOSE: value = calculateValue(data.getClose(),0); break;
            case LOW: value = calculateValue(data.getLow(),0); break;
            case HIGH: value = calculateValue(data.getHigh(),0); break;
            case VOLUME: value = calculateValue(data.getVolume(),0); break;
            default: throw new NoSuchElementException();
        }
        return value;
    }

    public int totalExamples() { return train.size() - exampleLength - predictLength; }

    public int inputColumns() { return VECTOR_SIZE; }

    @Override public int totalOutcomes() {
        if (this.category.equals(PriceCategory.ALL)) return VECTOR_SIZE;
        else return predictLength;
    }

    public boolean resetSupported() { return false; }

    public boolean asyncSupported() { return false; }

    public void reset() { initializeOffsets(); }

    public int batch() { return miniBatchSize; }

    public int cursor() { return totalExamples() - exampleStartOffsets.size(); }

    public int numExamples() { return totalExamples(); }

    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    public DataSetPreProcessor getPreProcessor() { throw new UnsupportedOperationException("Not Implemented"); }

    public List<String> getLabels() { throw new UnsupportedOperationException("Not Implemented"); }

    public boolean hasNext() { return exampleStartOffsets.size() > 0; }

    public DataSet next() { return next(miniBatchSize); }
    
    private List<Pair<INDArray, INDArray>> generateTestDataSet (List<StockData> stockDataList) {
    	int window = exampleLength + predictLength;
    	List<Pair<INDArray, INDArray>> test = new ArrayList<>();
    	for (int i = 0; i < stockDataList.size() - window; i++) {
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');
    		for (int j = i; j < i + exampleLength; j++) {
    			StockData stock = stockDataList.get(j);
    			input.putScalar(new int[] {j - i, 0}, calculateValue(stock.getOpen(),0));
    			input.putScalar(new int[] {j - i, 1}, calculateValue(stock.getClose(),1));
    			input.putScalar(new int[] {j - i, 2}, calculateValue(stock.getLow(),2));
    			input.putScalar(new int[] {j - i, 3}, calculateValue(stock.getHigh(),3));

                if (VECTOR_SIZE == 5){
                    input.putScalar(new int[] {j - i, 4}, calculateValue(stock.getVolume(),4));
                }
    		}
            StockData stock = stockDataList.get(i + exampleLength);
            INDArray label;

            label = Nd4j.create(new int[] {1}, 'f');
            switch (category) {
                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
                    case VOLUME: label.putScalar(new int[] {0}, stock.getVolume()); break;
                    default: throw new NoSuchElementException();
            }

    		test.add(new Pair<>(input, label));
    	}
    	return test;
    }

	@SuppressWarnings("resource")
	private List<StockData> readStockDataFromFile (String filename) {
        List<StockData> stockDataList = new ArrayList<>();
        try {
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
        } catch (IOException e) {
            e.printStackTrace();
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




    private double calculateValue(double value, int index){
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
        System.out.println("medianNormalization value of "+value+"="+V1);
        System.out.println("denorm "+V1+"= "+(V1 * median[index] ));
       return V1;
    }

    private double tanhestimators(double value, int index){
        double V1 = 0.5 * (  Math.tanh(0.01 * (value - minArray[index]) /  maxArray[index] ) + 1 );
        return V1;

    }

    public double detanhestimators(double value,int index){
        //   System.out.println("df");
        return EvaluationMatrix.deTanh(value,stvArray[index],meanArray[index]);
    }



//    atanh.value( data / 0.5  - 1) / 0.01 * (sdt + mean);

}
