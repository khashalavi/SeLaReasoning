How to run it

``` 
    cd ~/SeLaReasoning
    ./Sela_beat.sh --plan      # preview: 35 runs → 84 GPU jobs (main phase Qwen3-8B, all 5 datasets)
    sbatch Sela_beat.sh        # the heart beat
    ./reporting.sh             # → result/selar_report.md (also written automatically after every pass)
```

