package com.sronglong.crypto.representation;

/**
 * Created by zhanghao on 26/7/17.
 * Modifired by Sronglong
 * @author ZHANG HAO
 */
public class StockData {
    private String date; // date
    private String symbol; // stock name

    private double open; // open price
    private double close; // close price
    private double low; // low price
    private double high; // high price
    private double volume; // volume
    private double btc; // btc

    public StockData () {}

    public StockData (String date, String symbol, double open, double close, double low, double high, double volume,double btc) {
        this.date = date;
        this.symbol = symbol;
        this.open = open;
        this.close = close;
        this.low = low;
        this.high = high;
        this.volume = volume;
        this.btc = btc;
    }

    public String getDate() { return date; }
    public void setDate(String date) { this.date = date; }

    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }

    public double getOpen() { return open; }
    public void setOpen(double open) { this.open = open; }

    public double getClose() { return close; }
    public void setClose(double close) { this.close = close; }

    public double getLow() { return low; }
    public void setLow(double low) { this.low = low; }

    public double getHigh() { return high; }
    public void setHigh(double high) { this.high = high; }

    public double getVolume() { return volume; }
    public void setVolume(double volume) { this.volume = volume; }

    public double getBtc(){
        return  btc;
    }
    public void setBtc(double btc){this.volume = volume;}
}
