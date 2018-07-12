package model;

import java.util.ArrayList;

public class Matrix<E> {

    ArrayList<ArrayList<E>> matrix;
    Integer rows,columns;

    public Matrix(Integer rows, Integer columns, E initElem){
        this.columns = columns;
        this.rows = rows;
        matrix = new ArrayList<>();
        initMatrix(initElem);
    }

    private void initMatrix(E elem){
        for(int i=0;i<rows;i++){
            ArrayList<E> row = new ArrayList<>();
            for(int j=0;j<columns;j++){
                row.add(elem);
            }
            matrix.add(row);
        }
    }

    public E get(Integer row,Integer column){
        return matrix.get(row).get(column);
    }


    public void set(E value,Integer row,Integer column){
        matrix.get(row).set(column,value);
    }

    public Integer getRows() {
        return rows;
    }

    public Integer getColumns() {
        return columns;
    }

    public void setRows(Integer rows) {
        this.rows = rows;
    }

    public ArrayList<E> getRow(Integer index){
        return matrix.get(index);
    }

    public void setRow(ArrayList<E> row,Integer index){
        matrix.set(index,row);
    }

}
