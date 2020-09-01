call plug#begin('~/.local/share/nvim/plugged')

Plug 'morhetz/gruvbox'

Plug 'tmhedberg/SimpylFold'

Plug 'machakann/vim-highlightedyank'
hi HighlightedyankRegion cterm=reverse gui=reverse

Plug 'terryma/vim-multiple-cursors'

Plug 'neomake/neomake'
let g:neomake_python_enabled_makers = ['pylint']

Plug 'preservim/nerdtree'

Plug 'davidhalter/jedi-vim'
" disable autocompletion, cause we use deoplete for completion
let g:jedi#completions_enabled = 0

" open the go-to function in split, not another buffer
let g:jedi#use_splits_not_buffers = "right"

Plug 'sbdchd/neoformat'
" Enable alignment
let g:neoformat_basic_format_align = 1

" Enable tab to spaces conversion
let g:neoformat_basic_format_retab = 1

" Enable trimmming of trailing whitespace
let g:neoformat_basic_format_trim = 1

Plug 'jiangmiao/auto-pairs'

Plug 'preservim/nerdcommenter'

Plug 'https://github.com/Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
let g:deoplete#enable_at_startup = 1
let g:python2_host_prog = "/usr/bin/python2"
let g:python3_host_prog = "/usr/bin/python3"

Plug 'deoplete-plugins/deoplete-jedi'
autocmd InsertLeave,CompleteDone * if pumvisible() == 0 | silent! pclose! | endif
set splitbelow
inoremap <expr><tab> pumvisible() ? "\<c-n>" : "\<tab>"

Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
let g:airline_theme='bubblegum'

call plug#end()

" Full config: when writing or reading a buffer, and on changes in insert and
" normal mode (after 500ms; no delay when writing)
call neomake#configure#automake('nrwi', 500)

colorscheme gruvbox
set background=dark " use dark mode
" set background=light " uncomment to use light mode
