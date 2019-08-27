package com.isaac.stock.representation;

import com.google.common.collect.ImmutableMap;
import com.isaac.stock.predict.StockPricePrediction;
import com.isaac.stock.utils.EvaluationMatrix;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math3.analysis.function.Atanh;
import org.apache.commons.math3.analysis.function.Tanh;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 * @author ZHANG HAO
 */
public class CryptoDataSetIterator implements DataSetIterator {
    private static final Logger log = LoggerFactory.getLogger(StockPricePrediction.class);
    /** category and its index */
//    , PriceCategory.CLOSE, 1,
//    PriceCategory.LOW, 2, PriceCategory.HIGH, 3,PriceCategory.VOLUME,4 ,PriceCategory.BTC,5
    private final Map<PriceCategory, Integer> featureMapIndex = ImmutableMap.of(
            PriceCategory.OPEN, 0,
            PriceCategory.CLOSE, 1,
            PriceCategory.LOW, 2,
            PriceCategory.HIGH,3,
            PriceCategory.VOLUME,4);

//    public double value_mean;
//    public double value_deviation;
    public Tanh tanh = new Tanh();

    private final int VECTOR_SIZE = 4; // number of features for a stock data
    private int miniBatchSize; // mini-batch size
    private int exampleLength = 22; // default 22, say, 22 working days per month
    private int predictLength = 1; // default 1, say, one day ahead prediction

    /** minimal values of each feature in stock dataset */
    private double[] minArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] maxArray = new double[VECTOR_SIZE];

    /** feature to be selected as a training target */
    private PriceCategory category;

    /** mini-batch offset */
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    /** stock dataset for training */
    private List<StockData> train;
    /** adjusted stock dataset for testing */
    private List<Pair<INDArray, INDArray>> test;


    public CryptoDataSetIterator(String filename, String symbol, int miniBatchSize, int exampleLength, double splitRatio, PriceCategory category) {
        List<StockData> stockDataList = readDataFromFile(filename);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(stockDataList.size() * splitRatio);
        train = stockDataList.subList(0, split);
        test = generateTestDataSet(stockDataList.subList(split, stockDataList.size()));
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


    @Override
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label;
        if (category.equals(PriceCategory.ALL)) label = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        else label = Nd4j.create(new int[] {actualMiniBatchSize, predictLength, exampleLength}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            StockData curData = train.get(startIdx);
            StockData nextData;
            for (int i = startIdx; i < endIdx; i++) {
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, 0.5 * ( tanh.value(0.01 * (curData.getOpen() - minArray[0]) /  maxArray[0] ) + 1));
                input.putScalar(new int[] {index, 1, c}, 0.5 * ( tanh.value(0.01 * (curData.getClose() - minArray[1]) /  maxArray[1] ) + 1));
                input.putScalar(new int[] {index, 2, c}, 0.5 * ( tanh.value(0.01 * (curData.getLow() - minArray[2]) /  maxArray[2] ) + 1));
                input.putScalar(new int[] {index, 3, c}, 0.5 * ( tanh.value(0.01 * (curData.getHigh() - minArray[3]) /  maxArray[3] ) + 1));
              //  input.putScalar(new int[] {index, 4, c}, 0.5 * ( tanh.value(0.01 * (curData.getVolume() - value_mean) /  value_deviation ) + 1));
//                input.putScalar(new int[] {index, 5, c}, (curData.getBtc() - minArray[5]) / (maxArray[5] - minArray[5]));
                nextData = train.get(i + 1);
                if (category.equals(PriceCategory.ALL)) {
                    label.putScalar(new int[] {index, 0, c}, 0.5 * ( tanh.value(0.01 * (nextData.getOpen() - minArray[0]) /  maxArray[0] ) + 1));
                    label.putScalar(new int[] {index, 1, c}, 0.5 * ( tanh.value(0.01 * (nextData.getClose() - minArray[1]) /  maxArray[1] ) + 1));
                    label.putScalar(new int[] {index, 2, c}, 0.5 * ( tanh.value(0.01 * (nextData.getLow() - minArray[2]) /  maxArray[2] ) + 1));
                    label.putScalar(new int[] {index, 3, c}, 0.5 * ( tanh.value(0.01 * (nextData.getHigh() - minArray[3]) /  maxArray[3] ) + 1));
                 //   label.putScalar(new int[] {index, 4, c}, 0.5 * ( tanh.value(0.01 * (nextData.getVolume() - value_mean) /  value_deviation ) + 1));
//                    input.putScalar(new int[] {index, 5, c}, (nextData.getBtc() - minArray[5]) / (maxArray[5] - minArray[5]));
                } else {
                    label.putScalar(new int[]{index, 0, c}, feedLabel(nextData));
                }
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
//        log.info("train batch" );
     //   log.info(input.toString());
      //  log.info("label" + label.toString());

        return new DataSet(input, label);
    }

    private double feedLabel(StockData data) {
        double value;

        switch (category) {
            case OPEN: value = 0.5 * ( tanh.value(0.01 * (data.getOpen() - minArray[0]) /  maxArray[0] ) + 1); break;
            case CLOSE: value = 0.5 * ( tanh.value(0.01 * (data.getClose() -  minArray[1]) /  maxArray[1] ) + 1); break;
            case LOW: value = 0.5 * ( tanh.value(0.01 * (data.getLow() -  minArray[2]) /  maxArray[2] ) + 1); break;
            case HIGH: value = 0.5 * ( tanh.value(0.01 * (data.getHigh() -  minArray[3]) /  maxArray[3] ) + 1); break;
//            case VOLUME: value = 0.5 * ( tanh.value(0.01 * (data.getVolume() - value_mean) /  maxArray[0] ) + 1); break;
//            case BTC: value = (data.getBtc() - minArray[5]) / (maxArray[5] - minArray[5]); break;
            default: throw new NoSuchElementException();
        }
        return value;
    }

    @Override public int totalExamples() { return train.size() - exampleLength - predictLength; }

    @Override public int inputColumns() { return VECTOR_SIZE; }

    @Override public int totalOutcomes() {
        if (this.category.equals(PriceCategory.ALL)) return VECTOR_SIZE;
        else return predictLength;
    }

    @Override public boolean resetSupported() { return false; }

    @Override public boolean asyncSupported() { return false; }

    @Override public void reset() { initializeOffsets(); }

    @Override public int batch() { return miniBatchSize; }

    @Override public int cursor() { return totalExamples() - exampleStartOffsets.size(); }

    @Override public int numExamples() { return totalExamples(); }

    @Override public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override public DataSetPreProcessor getPreProcessor() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override public List<String> getLabels() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override public boolean hasNext() { return exampleStartOffsets.size() > 0; }

    @Override public DataSet next() { return next(miniBatchSize); }
    
    private List<Pair<INDArray, INDArray>> generateTestDataSet (List<StockData> stockDataList) {
    	int window = exampleLength + predictLength;
        Tanh tanh = new Tanh();
    	List<Pair<INDArray, INDArray>> test = new ArrayList<>();
    	for (int i = 0; i < stockDataList.size() - window; i++) {
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');
    		for (int j = i; j < i + exampleLength; j++) {
    			StockData stock = stockDataList.get(j);

                input.putScalar(new int[] {j - i, 0}, 0.5 * ( tanh.value(0.01 * (stock.getOpen() - minArray[0]) /  maxArray[0] ) + 1 ));
                input.putScalar(new int[] {j - i, 1}, 0.5 * ( tanh.value(0.01 * (stock.getClose() - minArray[1]) /  maxArray[0] ) + 1 ));
    			input.putScalar(new int[] {j - i, 2}, 0.5 * ( tanh.value(0.01 * (stock.getLow() - minArray[2]) /  maxArray[0] ) + 1 ));
    			input.putScalar(new int[] {j - i, 3}, 0.5 * ( tanh.value(0.01 * (stock.getHigh() - minArray[3]) /  maxArray[0] ) + 1 ));
    		//	input.putScalar(new int[] {j - i, 4}, 0.5 * ( tanh.value(0.01 * (stock.getVolume() - value_mean) /  value_deviation ) + 1 ));

    		}
            StockData stock = stockDataList.get(i + exampleLength);
            INDArray label;
            if (category.equals(PriceCategory.ALL)) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f'); // ordering is set as 'f', faster construct
                label.putScalar(new int[] {0}, stock.getOpen());
                label.putScalar(new int[] {1}, stock.getClose());
                label.putScalar(new int[] {2}, stock.getLow());
                label.putScalar(new int[] {3}, stock.getHigh());
              //  label.putScalar(new int[] {4}, stock.getVolume());
//                label.putScalar(new int[] {5}, stock.getBtc());
            } else {
                label = Nd4j.create(new int[] {1}, 'f');
                switch (category) {
                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
                    case VOLUME: label.putScalar(new int[] {0}, stock.getVolume()); break;
//                    case BTC: label.putScalar(new int[] {0}, stock.getBtc()); break;
                    default: throw new NoSuchElementException();
                }
            }
    		test.add(new Pair<>(input, label));
    	}
    	return test;
    }

	private List<StockData> readDataFromFile (String filename) {
        List<StockData> stockDataList = new ArrayList<>();
        try {

            List<String[]> list = new CSVReader(new FileReader(filename)).readAll(); // load all elements in a list
            int i = 0;
            for (String[] arr : list) {
                //skip first header.
               if (!arr[2].equals("open")){
                   double[] nums = new double[VECTOR_SIZE];
//                   Double.valueOf(arr[i + 2])
                   stockDataList.add(new StockData(arr[0], arr[1],Double.valueOf(arr[2]), Double.valueOf(arr[3]), Double.valueOf(arr[4]), Double.valueOf(arr[5]), 0,0));
               }
//               i++;
//               if (i > 5000){
//                   break;
//               }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


        TanHNormalizer(stockDataList);

        return stockDataList;
    }


    private void calculateParameters(double[] input, int index) {

        Mean mean = new Mean();

        minArray[index] = mean.evaluate(input); // mean
        maxArray[index] = new StandardDeviation().evaluate(input);

    }

    public double[] denormalize(double[] data ) {

        Atanh atanh = new Atanh();
        double[] res = new double[data.length];

        for (int i=0; i<data.length; i++) {
//            res[i] = atanh.value( data[i] / 0.5  - 1) / 0.01 * value_deviation + value_mean;
        }

        return res;

    }


    /**
     * Calculate Tanh estimator mean and deviation values for dataset values
     * @param dataset List of TrainDataItems
     */
    public void TanHNormalizer(List<StockData> dataset) {

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
        log.info(minArray.toString());
        log.info(maxArray.toString());
        Arrays.stream(minArray).forEach(System.out::println);
        Arrays.stream(maxArray).forEach(System.out::println);
    }

}
