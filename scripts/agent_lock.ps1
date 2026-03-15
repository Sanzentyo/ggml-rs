param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
cargo +nightly -Zscript "$ScriptDir/agent_lock.rs" -- @Args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
