import pandas as pd 
import pathlib
from openpyxl import load_workbook

def add_contrast_labels(dir) :
    dir = pathlib.Path(dir).rglob("*csv")
    #https://stackoverflow.com/questions/49681392/python-pandas-how-to-write-in-a-specific-column-in-an-excel-sheet

    for file in dir :
        writer = pd.ExcelWriter("contrast_labels.xlsx", engine="openpyxl")
        wb = writer.book
        df = pd.read_csv(file)
        output_col = []
        for gold, pred, clause, contrast_bin in zip(df["Gold"], df["predictions"], df["Clause"], df["Contrast"]) :
            if gold == "STATIVE" :
                output_col.append("DYNAMIC")
            elif gold == "DYNAMIC" :
                output_col.append("STATIVE")
            else :
                output_col.append("CANNOT_DECIDE")
        output_df = pd.DataFrame({"Contrast_Labels" : output_col})
        output_df.to_excel(writer, index=False)
        wb.save(file.name + "_" + "contrast_labels.xlsx")

if __name__ == "__main__":
    dir = "sd_contrast_output"
    add_contrast_labels(dir)