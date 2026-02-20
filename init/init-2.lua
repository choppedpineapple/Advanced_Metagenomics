-- =============================================================================
-- Neovim Configuration
-- No external plugins. No LSP. Pure offline bliss.
-- =============================================================================


-- =============================================================================
-- GENERAL OPTIONS
-- =============================================================================

local opt = vim.opt

-- Line numbers: absolute on current line, relative everywhere else
opt.number         = true
opt.relativenumber = true

-- Tabs & indentation (4 spaces, no tabs in the file)
opt.tabstop        = 4        -- a <Tab> in the file looks like 4 spaces
opt.shiftwidth     = 4        -- >> and << shift by 4 spaces
opt.softtabstop    = 4        -- <Tab> in insert mode inserts 4 spaces
opt.expandtab      = true     -- convert tabs to spaces
opt.smartindent    = true     -- auto-indent new lines based on context
opt.autoindent     = true     -- copy indent from current line on newline

-- Searching
opt.hlsearch       = false    -- don't keep highlighting after search is done
opt.incsearch      = true     -- highlight matches as you type
opt.ignorecase     = true     -- case-insensitive search...
opt.smartcase      = true     -- ...unless you type an uppercase letter

-- Appearance
opt.termguicolors  = true     -- enable 24-bit RGB colour (required for the theme below)
opt.cursorline     = true     -- highlight the line your cursor is on
opt.signcolumn     = "yes"    -- always show the sign column so text doesn't jump
opt.colorcolumn    = "88"     -- vertical ruler at column 88 (PEP-8 extended limit)
opt.wrap           = false    -- don't wrap long lines
opt.scrolloff      = 8        -- keep 8 lines visible above/below the cursor
opt.sidescrolloff  = 8        -- keep 8 columns visible left/right of the cursor
opt.showmode       = false    -- don't show -- INSERT -- etc. in the cmdline
opt.cmdheight      = 1        -- one line for the command bar
opt.pumheight      = 10       -- max items shown in the completion popup

-- Splits open below and to the right (natural reading direction)
opt.splitbelow     = true
opt.splitright     = true

-- Buffers & files
opt.hidden         = true     -- allow switching buffers without saving
opt.swapfile       = false    -- no .swp files cluttering your projects
opt.backup         = false    -- no backup files either
opt.undofile       = true     -- persistent undo across sessions (stored in undodir)
opt.undodir        = vim.fn.stdpath("data") .. "/undo"

-- Performance
opt.updatetime     = 250      -- faster CursorHold events (default 4000 is ancient)
opt.timeoutlen     = 400      -- how long to wait for a mapped key sequence

-- Clipboard: sync with system clipboard
opt.clipboard      = "unnamedplus"

-- Completion behaviour in command mode
opt.wildmode       = "longest:full,full"
opt.wildignorecase = true

-- Better display of whitespace
opt.list           = true
opt.listchars      = { tab = "→ ", trail = "·", nbsp = "␣" }

-- Fold settings (manual by default, no auto-fold on open)
opt.foldmethod     = "indent"
opt.foldlevel      = 99       -- start with everything unfolded

-- Mouse support (yes, even pros use it sometimes)
opt.mouse          = "a"

-- Encoding
opt.encoding       = "utf-8"
opt.fileencoding   = "utf-8"


-- =============================================================================
-- LEADER KEY
-- =============================================================================

-- Set leader to space before any mappings so everything picks it up correctly
vim.g.mapleader      = " "
vim.g.maplocalleader = " "


-- =============================================================================
-- FILETYPE-SPECIFIC SETTINGS
-- =============================================================================

-- Create the undo directory if it doesn't exist yet
vim.fn.mkdir(vim.fn.stdpath("data") .. "/undo", "p")

local filetype_group = vim.api.nvim_create_augroup("FiletypeSettings", { clear = true })

-- Python: PEP-8 style (already covered by globals, but explicit is better)
vim.api.nvim_create_autocmd("FileType", {
    group   = filetype_group,
    pattern = "python",
    callback = function()
        vim.opt_local.tabstop     = 4
        vim.opt_local.shiftwidth  = 4
        vim.opt_local.expandtab   = true
        vim.opt_local.colorcolumn = "88"  -- black formatter limit
    end,
})

-- Bash / shell
vim.api.nvim_create_autocmd("FileType", {
    group   = filetype_group,
    pattern = { "sh", "bash", "zsh" },
    callback = function()
        vim.opt_local.tabstop    = 4
        vim.opt_local.shiftwidth = 4
        vim.opt_local.expandtab  = true
    end,
})

-- C and C++: common style uses 4 spaces
vim.api.nvim_create_autocmd("FileType", {
    group   = filetype_group,
    pattern = { "c", "cpp" },
    callback = function()
        vim.opt_local.tabstop     = 4
        vim.opt_local.shiftwidth  = 4
        vim.opt_local.expandtab   = true
        vim.opt_local.colorcolumn = "100"
    end,
})

-- Rust: rustfmt default is 4 spaces, 100 col limit
vim.api.nvim_create_autocmd("FileType", {
    group   = filetype_group,
    pattern = "rust",
    callback = function()
        vim.opt_local.tabstop     = 4
        vim.opt_local.shiftwidth  = 4
        vim.opt_local.expandtab   = true
        vim.opt_local.colorcolumn = "100"
    end,
})

-- Remove trailing whitespace on save (for the filetypes we care about)
vim.api.nvim_create_autocmd("BufWritePre", {
    group   = filetype_group,
    pattern = { "*.py", "*.sh", "*.bash", "*.c", "*.cpp", "*.h", "*.rs" },
    callback = function()
        local pos = vim.api.nvim_win_get_cursor(0)
        vim.cmd([[%s/\s\+$//e]])
        vim.api.nvim_win_set_cursor(0, pos)
    end,
})

-- Highlight the text you yanked briefly so you know what was copied
vim.api.nvim_create_autocmd("TextYankPost", {
    group    = filetype_group,
    callback = function()
        vim.highlight.on_yank({ higroup = "IncSearch", timeout = 150 })
    end,
})


-- =============================================================================
-- KEYMAPS
-- =============================================================================

local map = function(mode, lhs, rhs, opts)
    opts = opts or {}
    opts.silent = opts.silent ~= false  -- silent by default
    vim.keymap.set(mode, lhs, rhs, opts)
end

-- Move between windows with Ctrl+hjkl (no need for Ctrl-W prefix)
map("n", "<C-h>", "<C-w>h")
map("n", "<C-j>", "<C-w>j")
map("n", "<C-k>", "<C-w>k")
map("n", "<C-l>", "<C-w>l")

-- Resize windows with arrow keys
map("n", "<C-Up>",    "<cmd>resize +2<CR>")
map("n", "<C-Down>",  "<cmd>resize -2<CR>")
map("n", "<C-Left>",  "<cmd>vertical resize -2<CR>")
map("n", "<C-Right>", "<cmd>vertical resize +2<CR>")

-- Move lines up/down in visual mode (like VS Code Alt+j/k)
map("v", "J", ":m '>+1<CR>gv=gv")
map("v", "K", ":m '<-2<CR>gv=gv")

-- Keep cursor centred when jumping around
map("n", "<C-d>", "<C-d>zz")
map("n", "<C-u>", "<C-u>zz")
map("n", "n",     "nzzzv")
map("n", "N",     "Nzzzv")
map("n", "G",     "Gzz")

-- Paste over selection without clobbering the default register
map("x", "<leader>p", [["_dP]])

-- Delete to black hole register (don't put deleted text into clipboard)
map({ "n", "v" }, "<leader>d", [["_d]])

-- Clear search highlight
map("n", "<Esc>", "<cmd>nohlsearch<CR>")

-- Save with Ctrl+S (works in normal and insert mode)
map("n", "<C-s>", "<cmd>w<CR>")
map("i", "<C-s>", "<Esc><cmd>w<CR>")

-- Quit window
map("n", "<leader>q", "<cmd>q<CR>")

-- Buffer navigation
map("n", "<S-h>", "<cmd>bprevious<CR>")
map("n", "<S-l>", "<cmd>bnext<CR>")
map("n", "<leader>bd", "<cmd>bdelete<CR>")

-- Open netrw file explorer (built-in, no plugins needed)
map("n", "<leader>e", "<cmd>Explore<CR>")

-- Indent/dedent and stay in visual mode
map("v", "<", "<gv")
map("v", ">", ">gv")

-- Select all
map("n", "<leader>a", "ggVG")

-- Quick vertical/horizontal splits
map("n", "<leader>sv", "<cmd>vsplit<CR>")
map("n", "<leader>sh", "<cmd>split<CR>")

-- Toggle line wrap
map("n", "<leader>tw", "<cmd>set wrap!<CR>")

-- Toggle relative numbers (useful when copy-pasting from someone else's screen)
map("n", "<leader>tr", "<cmd>set relativenumber!<CR>")


-- =============================================================================
-- NETRW (built-in file browser) SETTINGS
-- =============================================================================

vim.g.netrw_banner    = 0        -- hide the top banner
vim.g.netrw_liststyle = 3        -- tree view
vim.g.netrw_winsize   = 25       -- width when opened as a split


-- =============================================================================
-- CUSTOM COLOURSCHEME: "Dusk"
-- A dim, low-contrast blue-purple theme. All defined inline — no plugins.
-- =============================================================================

-- Colour palette
local c = {
    -- Backgrounds
    bg          = "#16172b",   -- main background
    bg_dark     = "#101120",   -- darker areas (e.g. sidebar)
    bg_popup    = "#1c1d32",   -- floating windows / popup menus
    bg_select   = "#232440",   -- visual selection, highlighted lines
    bg_line     = "#1a1b2f",   -- cursor line
    bg_col      = "#1e1f34",   -- colour column

    -- Foregrounds
    fg          = "#9ba8c9",   -- default text
    fg_dim      = "#636b8a",   -- dimmed text (line numbers, etc.)
    fg_dark     = "#4a5070",   -- very dim (indent guides, borders)

    -- Syntax colours (all muted and low contrast)
    blue        = "#6d8fc7",   -- keywords, function names
    blue_light  = "#7ea7d8",   -- builtins, methods
    purple      = "#9175c2",   -- type names, class definitions
    purple_soft = "#7b6baa",   -- constants, enum variants
    cyan        = "#5f9ea8",   -- strings
    teal        = "#4e8f8a",   -- special strings (regex, escape seqs)
    green       = "#5f8f6a",   -- numbers, booleans
    yellow      = "#a89060",   -- warnings, attributes
    orange      = "#b07a5a",   -- annotations, decorators
    red         = "#a05f6a",   -- errors, deletion
    red_soft    = "#8a5060",   -- dimmer red

    -- UI colours
    border      = "#2e3055",   -- window borders
    comment     = "#3d4468",   -- comments
    sign_add    = "#4a7060",   -- git add sign
    sign_change = "#5a6090",   -- git change sign
    sign_del    = "#8a4050",   -- git delete sign
    search_bg   = "#2d3560",   -- search highlight background
    match_bg    = "#3d2860",   -- other match highlight
}

-- Apply the theme
local function apply_theme()
    -- Reset everything to a clean state first
    vim.cmd("highlight clear")
    if vim.fn.exists("syntax_on") then
        vim.cmd("syntax reset")
    end
    vim.g.colors_name = "dusk"

    local hl = vim.api.nvim_set_hl

    -- -------------------------------------------------------------------------
    -- Editor chrome
    -- -------------------------------------------------------------------------
    hl(0, "Normal",          { fg = c.fg,        bg = c.bg })
    hl(0, "NormalNC",        { fg = c.fg_dim,    bg = c.bg_dark })     -- inactive windows
    hl(0, "NormalFloat",     { fg = c.fg,        bg = c.bg_popup })
    hl(0, "FloatBorder",     { fg = c.border,    bg = c.bg_popup })
    hl(0, "FloatTitle",      { fg = c.blue,      bg = c.bg_popup })

    hl(0, "CursorLine",      { bg = c.bg_line })
    hl(0, "CursorLineNr",    { fg = c.blue,      bg = c.bg_line,   bold = true })
    hl(0, "LineNr",          { fg = c.fg_dark })
    hl(0, "SignColumn",      { fg = c.fg_dark,   bg = c.bg })
    hl(0, "ColorColumn",     { bg = c.bg_col })

    hl(0, "VertSplit",       { fg = c.border,    bg = c.bg })
    hl(0, "WinSeparator",    { fg = c.border,    bg = c.bg })

    hl(0, "StatusLine",      { fg = c.fg,        bg = c.bg_popup })
    hl(0, "StatusLineNC",    { fg = c.fg_dark,   bg = c.bg_dark })
    hl(0, "TabLine",         { fg = c.fg_dim,    bg = c.bg_dark })
    hl(0, "TabLineFill",     { bg = c.bg_dark })
    hl(0, "TabLineSel",      { fg = c.fg,        bg = c.bg,         bold = true })

    hl(0, "EndOfBuffer",     { fg = c.bg_dark })
    hl(0, "Folded",          { fg = c.comment,   bg = c.bg_popup })
    hl(0, "FoldColumn",      { fg = c.fg_dark,   bg = c.bg })

    -- -------------------------------------------------------------------------
    -- Cursor & visual selection
    -- -------------------------------------------------------------------------
    hl(0, "Visual",          { bg = c.bg_select })
    hl(0, "VisualNOS",       { bg = c.bg_select })
    hl(0, "Cursor",          { fg = c.bg,        bg = c.blue })
    hl(0, "CursorIM",        { fg = c.bg,        bg = c.blue })
    hl(0, "TermCursor",      { fg = c.bg,        bg = c.blue })

    -- -------------------------------------------------------------------------
    -- Search
    -- -------------------------------------------------------------------------
    hl(0, "Search",          { fg = c.fg,        bg = c.search_bg })
    hl(0, "IncSearch",       { fg = c.bg,        bg = c.blue_light })
    hl(0, "Substitute",      { fg = c.bg,        bg = c.purple })
    hl(0, "MatchParen",      { bg = c.match_bg,  bold = true })

    -- -------------------------------------------------------------------------
    -- Messages & prompts
    -- -------------------------------------------------------------------------
    hl(0, "ModeMsg",         { fg = c.blue,      bold = true })
    hl(0, "MsgArea",         { fg = c.fg })
    hl(0, "MoreMsg",         { fg = c.cyan })
    hl(0, "Question",        { fg = c.blue })
    hl(0, "ErrorMsg",        { fg = c.red })
    hl(0, "WarningMsg",      { fg = c.yellow })

    -- -------------------------------------------------------------------------
    -- Popup menu (completion, wildmenu)
    -- -------------------------------------------------------------------------
    hl(0, "Pmenu",           { fg = c.fg,        bg = c.bg_popup })
    hl(0, "PmenuSel",        { fg = c.fg,        bg = c.bg_select,  bold = true })
    hl(0, "PmenuSbar",       { bg = c.bg_popup })
    hl(0, "PmenuThumb",      { bg = c.border })
    hl(0, "WildMenu",        { fg = c.fg,        bg = c.bg_select })

    -- -------------------------------------------------------------------------
    -- Diff
    -- -------------------------------------------------------------------------
    hl(0, "DiffAdd",         { fg = c.sign_add,    bg = "#1a2820" })
    hl(0, "DiffChange",      { fg = c.sign_change, bg = "#1a1e30" })
    hl(0, "DiffDelete",      { fg = c.sign_del,    bg = "#221820" })
    hl(0, "DiffText",        { fg = c.fg,          bg = "#252a48" })

    -- -------------------------------------------------------------------------
    -- Spelling
    -- -------------------------------------------------------------------------
    hl(0, "SpellBad",        { undercurl = true, sp = c.red })
    hl(0, "SpellCap",        { undercurl = true, sp = c.yellow })
    hl(0, "SpellLocal",      { undercurl = true, sp = c.cyan })
    hl(0, "SpellRare",       { undercurl = true, sp = c.purple })

    -- -------------------------------------------------------------------------
    -- Syntax groups (these map to the tree-sitter / regex highlights below)
    -- -------------------------------------------------------------------------
    hl(0, "Comment",         { fg = c.comment,      italic = true })
    hl(0, "Constant",        { fg = c.purple_soft })
    hl(0, "String",          { fg = c.cyan })
    hl(0, "Character",       { fg = c.teal })
    hl(0, "Number",          { fg = c.green })
    hl(0, "Float",           { fg = c.green })
    hl(0, "Boolean",         { fg = c.green,         bold = true })

    hl(0, "Identifier",      { fg = c.fg })
    hl(0, "Function",        { fg = c.blue_light })

    hl(0, "Statement",       { fg = c.blue })
    hl(0, "Conditional",     { fg = c.blue })
    hl(0, "Repeat",          { fg = c.blue })
    hl(0, "Label",           { fg = c.blue })
    hl(0, "Operator",        { fg = c.fg_dim })
    hl(0, "Keyword",         { fg = c.blue,         bold = true })
    hl(0, "Exception",       { fg = c.red })

    hl(0, "PreProc",         { fg = c.purple })
    hl(0, "Include",         { fg = c.purple })
    hl(0, "Define",          { fg = c.purple })
    hl(0, "Macro",           { fg = c.orange })
    hl(0, "PreCondit",       { fg = c.purple })

    hl(0, "Type",            { fg = c.purple })
    hl(0, "StorageClass",    { fg = c.blue })
    hl(0, "Structure",       { fg = c.purple })
    hl(0, "Typedef",         { fg = c.purple })

    hl(0, "Special",         { fg = c.teal })
    hl(0, "SpecialChar",     { fg = c.teal })
    hl(0, "Tag",             { fg = c.blue_light })
    hl(0, "Delimiter",       { fg = c.fg_dim })
    hl(0, "SpecialComment",  { fg = c.comment,      italic = true })
    hl(0, "Debug",           { fg = c.red })

    hl(0, "Underlined",      { underline = true })
    hl(0, "Ignore",          { fg = c.fg_dark })
    hl(0, "Error",           { fg = c.red,          bold = true })
    hl(0, "Todo",            { fg = c.yellow,       bold = true })

    -- -------------------------------------------------------------------------
    -- Tree-sitter highlights (Neovim 0.8+ @xxx.yyy namespace)
    -- These are the ones Neovim's bundled parsers actually use.
    -- -------------------------------------------------------------------------

    -- Variables & identifiers
    hl(0, "@variable",                  { fg = c.fg })
    hl(0, "@variable.builtin",          { fg = c.purple_soft,   italic = true })
    hl(0, "@variable.parameter",        { fg = c.fg })
    hl(0, "@variable.member",           { fg = c.fg })

    -- Constants
    hl(0, "@constant",                  { fg = c.purple_soft })
    hl(0, "@constant.builtin",          { fg = c.purple_soft,   bold = true })
    hl(0, "@constant.macro",            { fg = c.orange })

    -- Modules / namespaces
    hl(0, "@module",                    { fg = c.fg })
    hl(0, "@module.builtin",            { fg = c.purple_soft })
    hl(0, "@label",                     { fg = c.blue })

    -- Literals
    hl(0, "@string",                    { fg = c.cyan })
    hl(0, "@string.documentation",      { fg = c.cyan,          italic = true })
    hl(0, "@string.regexp",             { fg = c.teal })
    hl(0, "@string.escape",             { fg = c.teal,          bold = true })
    hl(0, "@string.special",            { fg = c.teal })
    hl(0, "@string.special.path",       { fg = c.cyan })
    hl(0, "@string.special.url",        { fg = c.blue_light,    underline = true })
    hl(0, "@string.special.symbol",     { fg = c.teal })

    hl(0, "@character",                 { fg = c.teal })
    hl(0, "@character.special",         { fg = c.teal })
    hl(0, "@boolean",                   { fg = c.green,         bold = true })
    hl(0, "@number",                    { fg = c.green })
    hl(0, "@number.float",              { fg = c.green })

    -- Types
    hl(0, "@type",                      { fg = c.purple })
    hl(0, "@type.builtin",              { fg = c.purple,        italic = true })
    hl(0, "@type.definition",           { fg = c.purple })
    hl(0, "@attribute",                 { fg = c.orange })
    hl(0, "@property",                  { fg = c.fg })

    -- Functions
    hl(0, "@function",                  { fg = c.blue_light })
    hl(0, "@function.builtin",          { fg = c.blue_light,    italic = true })
    hl(0, "@function.call",             { fg = c.blue_light })
    hl(0, "@function.macro",            { fg = c.orange })
    hl(0, "@function.method",           { fg = c.blue_light })
    hl(0, "@function.method.call",      { fg = c.blue_light })
    hl(0, "@constructor",               { fg = c.purple })

    -- Keywords
    hl(0, "@keyword",                   { fg = c.blue,         bold = true })
    hl(0, "@keyword.coroutine",         { fg = c.blue,         bold = true })
    hl(0, "@keyword.function",       
