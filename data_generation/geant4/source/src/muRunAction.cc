
#include "muRunAction.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"


muRunAction::muRunAction()
{}


muRunAction::~muRunAction()
{}


void muRunAction::BeginOfRunAction(const G4Run* aRun)
{ 
    G4cout << "### Run " << aRun->GetRunID() << " start." << G4endl;
    
    //inform the runManager to save random number seed
    G4RunManager::GetRunManager()->SetRandomNumberStore(true);
    
}


void muRunAction::EndOfRunAction(const G4Run* aRun)
{
    G4int NbOfEvents = aRun->GetNumberOfEvent();
    if (NbOfEvents == 0) return;
    G4cout << "### Run " << aRun->GetRunID() << " end." << G4endl;
    
    
}

