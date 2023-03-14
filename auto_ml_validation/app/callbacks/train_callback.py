@app.callback(
    Output('complete-train', 'children'),
    Input('submit', 'n_clicks'),
    State('train-data', 'data'),
    State('test-data', 'data'),
    State('target', 'value'),
    State('algo', 'value'),
    State('params', 'data'),
    prevent_initial_call=True
)
def train(n, train_data, test_data, target, algo, params):
    if n > 0:
        # convert jsonified data to df
        train_df = pd.read_json(train_data, orient='split')
        test_df = pd.read_json(test_data, orient='split')
        start = time.time()
        time.sleep(5)
        train_clf, train_metrics, test_clf, test_metrics = train_pipeline(
            train_df, test_df, target, algo, params
        )
        end = time.time()
        dur = (end - start)/60
        div = html.Div([
            html.P(f'Training completed! Time taken: {dur:.2f} mins.'),
            html.P('Performance on the training data:'),
            display_confusion_matrix(train_clf),
            display_dict(train_metrics),
            html.Br(),
            html.P('Performance on the testing data:'),
            display_confusion_matrix(test_clf),
            display_dict(test_metrics),
        ])
        return div
