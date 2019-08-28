package com.isaac.stock.utils;

import com.opencsv.CSVWriter;

import java.io.FileWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class CsvWriterExamples {

    public static String csvWriterOneByOne(List<String[]> stringArray, Path path) {
        try {
            CSVWriter writer = new CSVWriter(new FileWriter(path.toString()));
            for (String[] array : stringArray) {
                writer.writeNext(array);
            }
            writer.close();
        } catch (Exception ex) {
            Helpers.err(ex);
        }
        return Helpers.readFile(path);
    }

    public static String csvWriterAll(List<String[]> stringArray, Path path) {
        try {
            CSVWriter writer = new CSVWriter(new FileWriter(path.toString()));
            writer.writeAll(stringArray);
            writer.close();
        } catch (Exception ex) {
            Helpers.err(ex);
        }
//        return Helpers.readFile(path);
        return path.toAbsolutePath().toString();
    }

    public static List<String[]> toStringList(double[] actual, double[] predict, String name) {
        List<String[]> list = new ArrayList<>();
        list.add(new String[]{name, ""});
        list.add(new String[]{"actual", "predict"});
        for (int i = 0; i< actual.length ; i++){
            list.add(new String[]{String.valueOf(actual[i]),String.valueOf(predict[i]) });
        }
        return list;
    }
}