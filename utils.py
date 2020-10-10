def split_data(X,y,x_proportion = 0.9):
    '''
    We split data in X_train, y_train, X_test, y_test
    where X_train has balanced number of 0 label and 1 label
    
    return:
    X_train, y_train, X_test, y_test
    '''
    y = pd.DataFrame(y)
    y.columns = ['Default']
    positive_idx = y.index[y['Default'] == 1].tolist()
    negative_idx = y.index[y['Default'] == 0].tolist()
    
    # take 90% of positive label data into training set
    # take 600 of 0 label data into training set 
    
    pos_idx_split = int(len(positive_idx)*x_proportion)
    neg_idx_split = pos_idx_split + 100

    train_y_idx = positive_idx[:pos_idx_split] + negative_idx[:neg_idx_split]
    test_y_idx = positive_idx[pos_idx_split:] + negative_idx[neg_idx_split:]
    
    np.random.shuffle(train_y_idx)
    np.random.shuffle(test_y_idx)
    
    X_train = X.loc[train_y_idx]
    y_train = y['Default'][train_y_idx]
    X_test = X.loc[test_y_idx] 
    y_test = y['Default'][test_y_idx]
    
    return X_train,y_train,X_test,y_test


def impute_dataset_with_median(train_X,test_X):
    '''
    train_X:  training data features
    test_X:  test data features
    
    return:
    normalized_train_X: normalized traim data and with missing value imputed by column median 
    normalized_test_X: normalized test data and with missing value imputed by training data stats
    scalar: contain the scalar information for our trained dataset, use scalar.transform to transform test set and
    run evaluation
    '''
    # get nan value 
    nan = X['TDCyoy'][3]
    
    # fit the standard scalar with train data 
    scaler = StandardScaler()
    scaler.fit(train_X)
    # normalize train data 
    normalized_train_X = scaler.transform(train_X)
    normalized_train_X = pd.DataFrame(normalized_train_X)
    normalized_train_X.columns = train_X.columns
    
    # normalize test data using train data stats 
    normalized_test_X = scaler.transform(test_X)
    normalized_test_X = pd.DataFrame(normalized_test_X)
    normalized_test_X.columns = test_X.columns
    
    
    for col in normalized_train_X.columns:
        median_col = statistics.median(list(normalized_train_X[col]))
        if math.isnan(median_col):
            new_col = [i for i in normalized_train_X[col] if not math.isnan(i) ]
            median_col = statistics.median(new_col)
        normalized_train_X[col].fillna((median_col), inplace=True)
        normalized_test_X[col].fillna((median_col), inplace=True)
        

    return normalized_train_X,normalized_test_X, scaler
