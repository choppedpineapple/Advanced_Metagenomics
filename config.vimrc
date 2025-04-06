\}
let g:ale_fix_on_save = 1
let g:ale_python_flake8_options = '--max-line-length=120'
let g:ale_completion_enabled = 1

" Jedi-vim settings
let g:jedi#completions_enabled = 1
let g:jedi#enable_function_signature_help = 1

" Python PEP 8 indentation
let g:python_pep8_indent_hang = 1
let g:python_pep8_indent_multiline = 1

" Search settings
set ignorecase
set smartcase
set incsearch
set hlsearch

" Status line
set laststatus=2

" Terminal settings
" set termguicolors (removed, as this is related to color)

" Folding
set foldmethod=indent
set foldlevelstart=99

" Disable swap files
set noswapfile

" Disable backups
set nobackup
set nowritebackup
