# This file must be used with "source bin/activate.csh" *from csh*.
# You cannot run it directly.
# Created by Davide Di Blasi <davidedb@gmail.com>.

alias deactivate 'test $?_OLD_VIRTUAL_PATH != 0 && setenv PATH "$_OLD_VIRTUAL_PATH" && unset _OLD_VIRTUAL_PATH && test $?_OLD_VIRTUAL_PYTHONPATH != 0 && setenv PYTHONPATH "$_OLD_VIRTUAL_PYTHONPATH" && unset _OLD_VIRTUAL_PYTHONPATH; test $?_OLD_VIRTUAL_LD_LIBRARY_PATH != 0 && setenv LD_LIBRARY_PATH "$_OLD_VIRTUAL_LD_LIBRARY_PATH" && unset _OLD_VIRTUAL_LD_LIBRARY_PATH; rehash; test $?_OLD_VIRTUAL_PROMPT != 0 && set prompt="$_OLD_VIRTUAL_PROMPT" && unset _OLD_VIRTUAL_PROMPT; unsetenv VIRTUAL_ENV; test "\!:*" != "nondestructive" && unalias deactivate'

# Unset irrelavent variables.
deactivate nondestructive

setenv VIRTUAL_ENV "__DEDALUS_DIR__"

if ($?PATH == 0) then
    setenv PATH
endif
set _OLD_VIRTUAL_PATH="$PATH"
setenv PATH "${VIRTUAL_ENV}/bin:${PATH}"

if ($?PYTHONPATH == 0) then
    setenv PYTHONPATH
endif
set _OLD_VIRTUAL_PYTHONPATH="$PYTHONPATH"
setenv PYTHONPATH "${VIRTUAL_ENV}/lib/python3.4/site-packages:${VIRTUAL_ENV}/src/dedalus:${PYTHONPATH}"

if ($?LD_LIBRARY_PATH == 0) then
    setenv LD_LIBRARY_PATH
endif
set _OLD_VIRTUAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
setenv LD_LIBRARY_PATH "${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
### End extra yt vars

set _OLD_VIRTUAL_PROMPT="$prompt"

if ("" != "") then
    set env_name = ""
else
    if (`basename "$VIRTUAL_ENV"` == "__") then
        # special case for Aspen magic directories
        # see http://www.zetadev.com/software/aspen/
        set env_name = `basename \`dirname "$VIRTUAL_ENV"\``
    else
        set env_name = `basename "$VIRTUAL_ENV"`
    endif
endif
set prompt = "[$env_name] $prompt"
unset env_name

rehash
