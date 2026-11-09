
#pragma once

class TerminalUI
{
public:
    void run();

private:
    void displayMainMenu();

    void handleSingleGeneration();
    void handleBatchGeneration();

    void handleSolveExisting();

    void handleGenerateAndSolve();
};
