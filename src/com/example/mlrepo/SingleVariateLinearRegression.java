package com.example.mlrepo;

import java.util.ArrayList;

public class SingleVariateLinearRegression {

    private double w;
    private double b;

    SingleVariateLinearRegression()
    {
        this.w = 0.0;
        this.b = 0.0;
    }

    public double predict(double x)
    {
        return x*this.w + this.b;
    }


    private void update_w_and_b(ArrayList<Double> x, ArrayList<Double> y, double alpha)
    {
        double dl_dw = 0.0;
        double dl_db = 0.0;
        int N = x.size();

        for (int i=0;i<N;i++)
        {
            dl_dw += -2.0 * x.get(i) * (y.get(i) - predict(x.get(i)));
            dl_db += -2.0 * (y.get(i) - predict(i));
        }

        this.w = this.w - (double)(1/N)*dl_dw*alpha;
        this.b = this.b - (double)(1/N)*dl_db*alpha;
    }

    public double avg_loss(ArrayList<Double> x, ArrayList<Double> y)
    {
        double loss = 0.0;
        for (int i=0;i<x.size();i++)
        {
            loss += Math.pow(y.get(i) - predict(x.get(i)), 2.0);
        }
        return loss/x.size();
    }

    public void train(ArrayList<Double> x, ArrayList<Double> y, double alpha, int epochs, boolean logProgress)
    {
        for (int e=0;e<epochs;e++)
        {
            update_w_and_b(x,y,alpha);

            if (e == (int)(epochs*0.1) && logProgress==true)
            {
                System.out.println("epoch:"+e+" loss:"+avg_loss(x,y));
            }
        }
    }

}
