process_igblast_folders() {
    local root_dir="$PWD"
    local dir clean_dir igh igl

    pyenv_init() {
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    }
    
    pyenv_init
    pyenv activate edlib-env || { echo "Failed to activate pyenv environment!"; return 1; }

    for dir in "$root_dir"/*/; do
        [ -d "$dir" ] || continue
        
        clean_dir="${dir%/}"
        echo "Processing: $clean_dir"

        (
            cd "$dir" || { echo "Could not enter $dir. Skipping!"; exit 1; }
            
            igh="$PWD/igblast_heavy/igblast_heavy.tsv"
            igl="$PWD/igblast_light/igblast_light.tsv"

            if [[ ! -f "$igh" || ! -f "$igl" ]]; then
                echo "Warning: Missing TSV files in $clean_dir. Skipping!"
                exit 1
            fi

            time python3 "${root_dir}/igblast_output_parsing.py" "$igh" "$igl"
        )
        
        echo "Done with $clean_dir"
        echo "----------------------------------------"
    done

    pyenv deactivate
}
