
#ifndef muRunAction_h
#define muRunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"


class G4Run;

class muRunAction : public G4UserRunAction
{
public:
    muRunAction();
    ~muRunAction();
    
public:
    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);
    
private:
};


#endif

