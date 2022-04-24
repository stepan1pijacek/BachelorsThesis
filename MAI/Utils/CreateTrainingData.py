from MAI.Utils.ReadData.ReadData import main


def training_function():
    train_df, test_df, all_labels = main()
    # train_df['path'] = train_df['path'].astype(str)
    # train_df.rename(columns={'OriginalImagePixelSpacing[x,y] ': 'OriginalImagePixelSpacing[x,y]'})
    return train_df, test_df, all_labels
