from notears.test import (
    test_convergence,
    test_progress_rate,
    test_runtime,
    test_structure,
    test_general
)

if __name__ == '__main__':
    df_convergence = test_convergence()
    df_convergence.to_csv('./result/convergence.csv', index=False)
    #df_progress_rate = test_progress_rate()
    #df_progress_rate.to_csv('./result/progress_rate.csv', index=False)
    #df_runtime = test_runtime()
    #df_runtime.to_csv('./result/runtime.csv', index=False)
    #df_general = test_general()
    #df_general.to_csv('./result/general.csv', index=False)
    #for structure in ['fork', 'collider', 'chain', 'complex']:
    #    df1, df2 = test_structure(structure_type=structure)
    #    df1.to_csv(f'./result/{structure}_test_true.csv', index=False)
    #    df2.to_csv(f'./result/{structure}_test_count.csv', index=False)