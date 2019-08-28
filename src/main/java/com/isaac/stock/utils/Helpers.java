package com.isaac.stock.utils;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Helpers {

    /**
     * Write Files
     */

    public static Path fileOutAllPath() throws URISyntaxException {
        URI uri = ClassLoader.getSystemResource(Constants.CSV_All).toURI();
        return Paths.get(uri);
    }

    public static Path fileOutBeanPath() throws URISyntaxException {
        URI uri = ClassLoader.getSystemResource(Constants.CSV_BEAN).toURI();
        return Paths.get(uri);
    }

    public static Path fileOutOnePath(String name) throws IOException {
        File tmp = new File("src/main/resources/", "predict_result_".concat(String.valueOf(name)).concat(".csv"));
        tmp.createNewFile();
//        File locationToSave = new File("src/main/resources/predict_result_".concat(String.valueOf(name)).concat(".csv"));
//        URI uri = ClassLoader.getSystemResource(Constants.CSV_ONE).toURI();
//        locationToSave.mkdirs();
        return Paths.get(tmp.getAbsolutePath());
    }

    /**
     * Read Files
     */

    public static Path twoColumnCsvPath() throws URISyntaxException {
        URI uri = ClassLoader.getSystemResource(Constants.TWO_COLUMN_CSV).toURI();
        return Paths.get(uri);
    }

    public static Path fourColumnCsvPath() throws URISyntaxException {
        URI uri = ClassLoader.getSystemResource(Constants.FOUR_COLUMN_CSV).toURI();
        return Paths.get(uri);
    }

    public static Path namedColumnCsvPath() throws URISyntaxException {
        URI uri = ClassLoader.getSystemResource(Constants.NAMED_COLUMN_CSV).toURI();
        return Paths.get(uri);
    }

    /**
     * Simple File Reader
     */

    public static String readFile(Path path) {
        String response = "";
        try {
            FileReader fr = new FileReader(path.toString());
            BufferedReader br = new BufferedReader(fr);
            String strLine;
            StringBuffer sb = new StringBuffer();
            while ((strLine = br.readLine()) != null) {
                sb.append(strLine);
            }
            response = sb.toString();
            System.out.println(response);
            fr.close();
            br.close();
        } catch (Exception ex) {
            Helpers.err(ex);
        }
        return response;
    }

    /**
     * Dummy Data for Writing.
     */

    public static List<String[]> twoColumnCsvString() {
        List<String[]> list = new ArrayList<>();
        list.add(new String[]{"ColA", "ColB"});
        list.add(new String[]{"A", "B"});
        return list;
    }

    public static List<String[]> fourColumnCsvString() {
        List<String[]> list = new ArrayList<>();
        list.add(new String[]{"ColA", "ColB", "ColC", "ColD"});
        list.add(new String[]{"A", "B", "A", "B"});
        list.add(new String[]{"BB", "AB", "AA", "B"});
        return list;
    }

    /**
     * Message Helpers
     */

    public static void print(String msg) {
        System.out.println(msg);
    }

    public static void err(Exception ex) {
        System.out.println(Constants.GENERIC_EXCEPTION + " " + ex);
    }
}
