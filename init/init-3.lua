-- =============================================================================
-- NEOVIM CONFIGURATION
-- Standalone, Offline, No Plugins
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Leader Key Configuration
-- -----------------------------------------------------------------------------
-- Set the leader key to Space for easy access to custom shortcuts
vim.g.mapleader = " "
vim.g.maplocalleader = " "

-- -----------------------------------------------------------------------------
-- 2. General Editor Options
-- -----------------------------------------------------------------------------
local opt = vim.opt

-- Line numbering: Show absolute number on current line, relative on others
opt.number = true
opt.relativenumber = true

-- Indentation: 4 spaces, convert tabs to spaces (standard for Python/Rust/C)
opt.tabstop = 4
opt.shiftwidth = 4
opt.expandtab = true
opt.autoindent = true
opt.smartindent = true

-- Line wrapping: Do not wrap long lines automatically
opt.wrap = false

-- Cursor: Keep cursor visible during operations
opt.scrolloff = 8
opt.sidescrolloff = 8

-- Search: Ignore case unless capital letters are used
opt.ignorecase = true
opt.smartcase = true
opt.hlsearch = true
opt.incsearch = true

-- Appearance: Enable true colors and set background
opt.termguicolors = true
opt.background = "dark"
opt.signcolumn = "yes" -- Always show sign column (prevents layout shifts)

-- Performance & Safety
opt.updatetime = 250 -- Faster completion
opt.timeoutlen = 300 -- Faster key sequence recognition
opt.clipboard = "unnamedplus" -- Sync with system clipboardopt.backup = false -- Do not create backup files
opt.writebackup = false
opt.swapfile = false -- Disable swap files (prevents .swp clutter)
opt.undofile = true -- Enable persistent undo
opt.undolevels = 10000

-- -----------------------------------------------------------------------------
-- 3. Custom Colorscheme (Blueish-Purplish, Dim, Low-Contrast)
-- -----------------------------------------------------------------------------
-- Clear existing highlights to ensure a clean slate
vim.cmd("highlight clear")

-- Define color palette
local colors = {
    bg        = "#15151f", -- Deep dark blue/purple background
    fg        = "#8a8a9e", -- Dim grey foreground
    cursor    = "#8a8a9e", -- Cursor color
    line_nr   = "#3a3a4e", -- Low contrast line numbers
    active_ln = "#1f1f2e", -- Slightly lighter background for current line
    comment   = "#5a5a6e", -- Dim comments
    keyword   = "#6a7a9a", -- Muted blue for keywords
    string    = "#7a6a9a", -- Muted purple for strings
    func      = "#7a8a9a", -- Muted cyan/blue for functions
    type      = "#8a7a9a", -- Muted purple for types
    constant  = "#7a7a8a", -- Muted grey for constants
    error     = "#9a6a6a", -- Muted red for errors
    warning   = "#9a8a6a", -- Muted yellow for warnings
}

-- Helper function to set highlight groups
local function set_hl(group, fg, bg)
    local hl_opts = { fg = fg }
    if bg then
        hl_opts.bg = bg
    end
    vim.api.nvim_set_hl(0, group, hl_opts)
end

-- Apply colors to standard Vim highlight groups
set_hl("Normal", colors.fg, colors.bg)
set_hl("CursorLine", nil, colors.active_ln)
set_hl("LineNr", colors.line_nr, colors.bg)
set_hl("CursorLineNr", colors.fg, colors.active_ln)
set_hl("Comment", colors.comment)
set_hl("Keyword", colors.keyword)
set_hl("Statement", colors.keyword)
set_hl("String", colors.string)
set_hl("Constant", colors.constant)
set_hl("Function", colors.func)
set_hl("Identifier", colors.fg)set_hl("Type", colors.type)
set_hl("PreProc", colors.keyword)
set_hl("Special", colors.func)
set_hl("Error", colors.error)
set_hl("Todo", colors.warning)
set_hl("Visual", nil, colors.active_ln)
set_hl("Search", colors.bg, colors.keyword) -- Inverted search highlight
set_hl("IncSearch", colors.bg, colors.string)

-- Set the colorscheme name to prevent errors if Neovim tries to reload
vim.g.colors_name = "dim_purple_blue"

-- -----------------------------------------------------------------------------
-- 4. Syntax and Filetype Detection
-- -----------------------------------------------------------------------------
-- Enable syntax highlighting using built-in legacy syntax (No Treesitter)
vim.cmd("syntax enable")

-- Enable filetype detection, plugins, and indentation rules
-- This ensures Python, C, Rust, Bash get correct indentation and settings
vim.cmd("filetype plugin indent on")

-- -----------------------------------------------------------------------------
-- 5. Keybindings
-- -----------------------------------------------------------------------------
local keymap = vim.keymap

-- Save file
keymap.set("n", "<leader>w", "<cmd>w<cr>", { desc = "Save file" })

-- Quit Neovim
keymap.set("n", "<leader>q", "<cmd>q<cr>", { desc = "Quit Neovim" })

-- Save and Quit
keymap.set("n", "<leader>x", "<cmd>wq<cr>", { desc = "Save and Quit" })

-- Better window navigation (Ctrl + h/j/k/l)
keymap.set("n", "<C-h>", "<C-w>h", { desc = "Move to left window" })
keymap.set("n", "<C-j>", "<C-w>j", { desc = "Move to lower window" })
keymap.set("n", "<C-k>", "<C-w>k", { desc = "Move to upper window" })
keymap.set("n", "<C-l>", "<C-w>l", { desc = "Move to right window" })

-- Resize windows with arrows
keymap.set("n", "<C-Up>", "<cmd>resize +2<cr>", { desc = "Increase window height" })
keymap.set("n", "<C-Down>", "<cmd>resize -2<cr>", { desc = "Decrease window height" })
keymap.set("n", "<C-Left>", "<cmd>vertical resize -2<cr>", { desc = "Decrease window width" })
keymap.set("n", "<C-Right>", "<cmd>vertical resize +2<cr>", { desc = "Increase window width" })

-- Clear search highlighting after search
keymap.set("n", "<leader>n", "<cmd>nohlsearch<cr>", { desc = "Clear search highlight" })
-- Keep cursor centered when scrolling
keymap.set("n", "n", "nzzzv", { desc = "Next search result centered" })
keymap.set("n", "N", "Nzzzv", { desc = "Previous search result centered" })
keymap.set("n", "<C-d>", "<C-d>zz", { desc = "Half page down centered" })
keymap.set("n", "<C-u>", "<C-u>zz", { desc = "Half page up centered" })

-- -----------------------------------------------------------------------------
-- 6. Autocommands (Automatic Commands)
-- -----------------------------------------------------------------------------
local augroup = vim.api.nvim_create_augroup
local autocmd = vim.api.nvim_create_autocmd

-- Create a group for custom auto commands
local user_config = augroup("UserConfig", { clear = true })

-- Highlight current line only when active
autocmd({ "InsertLeave", "WinEnter" }, {
    group = user_config,
    pattern = "*",
    command = "set cursorline",
})
autocmd({ "InsertEnter", "WinLeave" }, {
    group = user_config,
    pattern = "*",
    command = "set nocursorline",
})

-- Remove trailing whitespace on save (Optional, keeps files clean)
autocmd("BufWritePre", {
    group = user_config,
    pattern = "*",
    command = "%s/\\s\\+$//e",
})

-- Return to last edit position when opening files
autocmd("BufReadPost", {
    group = user_config,
    pattern = "*",
    callback = function()
        local mark = vim.api.nvim_buf_get_mark(0, '"')
        local lcount = vim.api.nvim_buf_line_count(0)
        if mark[1] > 0 and mark[1] <= lcount then
            pcall(vim.api.nvim_win_set_cursor, 0, mark)
        end
    end,
})
