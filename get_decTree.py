def get_color_decTree(df, X_columns=None, y_columns=None, fname="tree"):
    X_columns = X_columns or [
        "APF", "XbC", "StdX", "YbC", "EGap","StdS"
    ]
    y_columns= y_columns or ["EHull"]
    X, y = get_data(
        df,
        X_columns=X_columns,
        y_columns=y_columns,
        # apply_function=True,
        normalize=False,
        test_size=0,
        nafill=0.0,
        expand_dimension=1)
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=6, max_features="log2")
    clf = clf.fit(X, y < 5e-2)
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=X_columns,
        filled=True,
        rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(fname)