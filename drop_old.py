import pandas as pd

ci = pd.read_csv('data/final/cleaned_issues.csv')
rl = pd.read_csv('data/final/final_rl_dataset.csv')

cols_to_drop = ['intent_group', 'confidence_score', 'uncertainty_flag',
                'confidence_band', 'rl_recommendation', 'suggested_action']

ci_dropped = [c for c in cols_to_drop if c in ci.columns]
rl_dropped = [c for c in cols_to_drop if c in rl.columns]

if ci_dropped:
    ci = ci.drop(columns=ci_dropped)
    ci.to_csv('data/final/cleaned_issues.csv', index=False, encoding='utf-8')
    print(f'Dropped from cleaned_issues.csv: {ci_dropped}')
else:
    print('cleaned_issues.csv — nothing to drop')

if rl_dropped:
    rl = rl.drop(columns=rl_dropped)
    rl.to_csv('data/final/final_rl_dataset.csv', index=False, encoding='utf-8')
    print(f'Dropped from final_rl_dataset.csv: {rl_dropped}')
else:
    print('final_rl_dataset.csv — nothing to drop')